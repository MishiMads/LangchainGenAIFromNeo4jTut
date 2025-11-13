from typing import Dict, Tuple, Optional, List
import numpy as np
from dataclasses import dataclass, field
from neo4j import GraphDatabase
import dotenv
import math, time, random
import pandas as pd
from datetime import datetime, timezone

# - Hyperparameters -#
GHOST_S = 1.0
GHOST_F = 1.0
ALPHA_PROPDEC = 0.5  # per-attempt decay toward 0.5 propdec
LEARNING_RATE = 0.01


# - Classes -#

@dataclass
class KCState:
    raw_suc: int = 0
    raw_fail: int = 0
    last_ts: Optional[datetime] = None
    streak: int = 0
    attempt_count: int = 0


@dataclass
class ItemState:
    last_ts: Optional[datetime] = None


@dataclass
class StudentState:
    pd_num: float = GHOST_S  # Ghost successes
    pd_den: float = GHOST_S + GHOST_F  # Ghost total
    last_ts: Optional[datetime] = None


class OnlineState:
    def __init__(self, alpha_propdec: float = ALPHA_PROPDEC):
        self.alpha_propdec = alpha_propdec
        self.kc_state: Dict[Tuple[int, int], KCState] = {}
        self.item_state: Dict[Tuple[int, int], ItemState] = {}
        self.stu_state: Dict[int, StudentState] = {}

    def _days_since(self, earlier: Optional[datetime], later: datetime) -> float:
        if earlier is None:
            return float('inf')
        return max((later - earlier).total_seconds() / 86400.0, 0.0)

    def snapshot_features(self, user: int, kc: int, item: int, now: datetime):
        stu = self.stu_state.get(user, StudentState())
        propdec_student = stu.pd_num / max(stu.pd_den, 1e-9)

        kcst = self.kc_state.get((user, kc), KCState())
        days_kc = self._days_since(kcst.last_ts, now)
        itemst = self.item_state.get((user, item), ItemState())
        days_item = self._days_since(itemst.last_ts, now)

        attempts_kc = kcst.raw_suc + kcst.raw_fail
        first_kc_attempt = (attempts_kc == 0)

        return {
            "propdec_student": propdec_student,
            "raw_suc_kc": kcst.raw_suc,
            "raw_fail_kc": kcst.raw_fail,
            "age_days_kc": days_kc,
            "age_days_item": days_item,
            "attempt_index_kc": attempts_kc,
            "first_kc_attempt": first_kc_attempt,
        }

    def update(self, user: int, kc: int, item: int, ts: datetime, outcome: int):
        # update student
        stu = self.stu_state.get(user, StudentState())
        stu.pd_num = self.alpha_propdec * stu.pd_num + float(outcome)
        stu.pd_den = self.alpha_propdec * stu.pd_den + 1.0
        stu.last_ts = ts
        self.stu_state[user] = stu

        # update KC
        kcst = self.kc_state.get((user, kc), KCState())
        if outcome == 1:
            kcst.raw_suc += 1
            kcst.streak += 1
        else:
            kcst.raw_fail += 1
            kcst.streak = 0
        kcst.last_ts = ts
        kcst.attempt_count += 1
        self.kc_state[(user, kc)] = kcst

        # update item
        itemst = self.item_state.get((user, item), ItemState())
        itemst.last_ts = ts
        self.item_state[(user, item)] = itemst


NEO4J_PASSWORD = dotenv.get_key(dotenv.find_dotenv(), "NEO4J_PASSWORD")

event_cols = ['user_id', 'item_id', 'kc_id', 'ts', 'outcome']
feat_cols = ['user_id', 'item_id', 'kc_id', 'ts_assigned', 'propdec_student_pre',
             'raw_suc_kc_pre', 'raw_fail_kc_pre', 'age_days_kc_pre', 'age_days_item_pre']
events = pd.DataFrame(columns=event_cols)
feature_log = pd.DataFrame(columns=feat_cols)


# -------------------------------------------------------------#

def get_all_kc_subject(tx):
    """
    Fetches the item catalog from the LangChain-generated graph.

    This query is now matched to your custom schema:
    - We treat 'Topic' as the "item" (e.g., a lesson or document section).
    - We treat 'Subject' as the "KC" (Knowledge Component) associated with that item.
    """
    q = """
    MATCH (item:Topic)-[:HAS_SUBJECT]->(kc:Subject)

    // Ensure the nodes have the 'id' property created by the transformer
    WHERE item.id IS NOT NULL AND kc.id IS NOT NULL

    RETURN DISTINCT
        id(item) AS item_id,
        item.id AS item_name,  // Use the 'id' property as the name
        id(kc) AS kc_id,
        kc.id AS kc_name       // Use the 'id' property as the name

    ORDER BY kc_name, item_name
    """
    return [dict(r) for r in tx.run(q)]

def snapshot_user_over_items(user_id: int, now: datetime, state: OnlineState,
                             item_catalog: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in item_catalog.iterrows():
        feats = state.snapshot_features(user_id, int(r.kc_id), int(r.item_id), now)
        rows.append({
            'user_id': user_id,
            'kc_id': int(r.kc_id),
            'kc_name': r.get('kc_name', ''),
            'item_id': int(r.item_id),
            'item_name': r.get('item_name', ''),
            'propdec_student_pre': feats['propdec_student'],
            'raw_suc_kc_pre': feats['raw_suc_kc'],
            'raw_fail_kc_pre': feats['raw_fail_kc'],
            'age_days_kc_pre': feats['age_days_kc'],
            'age_days_item_pre': feats['age_days_item'],
        })

        # print(rows[-1])  # Debug: print each row as it's created
    return pd.DataFrame(rows)


# --- Parameters for the LKT scoring ---
@dataclass
class LKTParams:
    b0: float = 0.5678346270290896  # global intercept
    w_propdec: float = -1.0396124740530723  # coef for propdec(student)
    w_linesuc: float = 1.2507228142790991  # coef for KC success count
    w_linefail: float = -1.1556277902622232  # coef for KC failure count
    w_recency_kc: float = -0.014934735713301214  # coef for recency feature
    recency_kc_shift: float = 0.5  # to avoid 0^(-d); paper suggests handling start carefully
    recency_kc_power: float = 0.25  # d in (age+shift)^(-d); learn offline
    # item intercepts (fixed effects). Fill from your fitted model.
    w_recency_item: float = 0.0
    recency_item_shift: float = 0.5
    recency_item_power: float = 0.25
    item_bias: Dict[int, float] = field(default_factory=dict)


LKT = LKTParams()


def record_outcome_and_update(user_id: int, kc_id: int, item_id: int, outcome: int,
                              state: OnlineState, ts: Optional[datetime] = None):
    ts = ts or datetime.now(timezone.utc).replace(microsecond=0)

    pre_event_features = state.snapshot_features(user_id, kc_id, item_id, ts)

    update_model_with_sgd(LKT, pre_event_features, item_id, outcome, LEARNING_RATE)

    global events
    events = pd.concat([events, pd.DataFrame([{
        'user_id': user_id, 'item_id': item_id, 'kc_id': kc_id,
        'ts': ts, 'outcome': int(outcome)
    }])], ignore_index=True)
    state.update(user_id, kc_id, item_id, ts, outcome)


def update_model_with_sgd(params: LKTParams, features: dict, item_id: int, outcome: int, learning_rate: float):
    b_item = params.item_bias.get(item_id, 0.0)
    rec_kc = recency_transform(float(features['age_days_kc']),
                               params.recency_kc_shift, params.recency_kc_power)
    rec_item = recency_transform(float(features['age_days_item']),
                                 params.recency_item_shift, params.recency_item_power)

    logit = (
            params.b0 + b_item
            + params.w_propdec * float(features['propdec_student'])
            + params.w_linesuc * float(features['raw_suc_kc'])
            + params.w_linefail * float(features['raw_fail_kc'])
            + params.w_recency_kc * rec_kc
            + params.w_recency_item * rec_item
    )
    p_hat = 1.0 / (1.0 + np.exp(-logit))

    error = p_hat - outcome

    params.b0 -= learning_rate * error * 1.0  # The feature for the intercept is always 1
    params.w_propdec -= learning_rate * error * float(features['propdec_student'])
    params.w_linesuc -= learning_rate * error * float(features['raw_suc_kc'])
    params.w_linefail -= learning_rate * error * float(features['raw_fail_kc'])
    params.w_recency_kc -= learning_rate * error * rec_kc
    params.w_recency_item -= learning_rate * error * rec_item

    # Update the item-specific bias
    item_id = item_id
    current_bias = params.item_bias.get(item_id, 0.0)
    params.item_bias[item_id] = current_bias - learning_rate * error * 1.0


def recency_transform(age_days: float, shift: float, power: float) -> float:
    # KC-level recency: power-law of time since last KC attempt.
    # First attempt: return 0 so it doesn’t give a spurious boost.
    if age_days <= 0.0:
        return 0.0
    return (age_days + shift) ** (-power)


def predict_lkt_logit_row(row: pd.Series, params: LKTParams) -> float:
    b_item = params.item_bias.get(int(row['item_id']), 0.0)
    rec_kc = recency_transform(float(row['age_days_kc_pre']),
                               params.recency_kc_shift, params.recency_kc_power)
    rec_item = recency_transform(float(row['age_days_item_pre']),
                                 params.recency_item_shift, params.recency_item_power)
    return (
            params.b0 + b_item
            + params.w_propdec * float(row['propdec_student_pre'])
            + params.w_linesuc * float(row['raw_suc_kc_pre'])
            + params.w_linefail * float(row['raw_fail_kc_pre'])
            + params.w_recency_kc * rec_kc
            + params.w_recency_item * rec_item  # breaks intra-KC ties
    )


def predict_lkt_p_hat(df: pd.DataFrame, params: LKTParams) -> np.ndarray:
    logits = df.apply(lambda r: predict_lkt_logit_row(r, params), axis=1).values
    return 1.0 / (1.0 + np.exp(-logits))


def select_next_item_lkt(
        user_id: int,
        now: datetime,
        state: OnlineState,
        item_catalog: pd.DataFrame,
        params: LKTParams = LKT,
        target_p: float = 0.80,  # “optimal difficulty” target, per paper examples
        mastery_p: float = 0.95,  # mastery gating threshold
        cooldown_kc_days: float = 0.0,  # optional spacing constraints
        cooldown_item_days: float = 0.0,
        epsilon: float = 0.1,  # ε-greedy exploration
        tie_tol: float = 0.02,  # tolerance for near-target p_hat
        jitter: float = 1e-6  # optional tiny jitter to avoid deterministic ties
) -> Optional[pd.Series]:
    snap = snapshot_user_over_items(user_id, now, state, item_catalog)
    if snap.empty:
        return None

    # Cooldowns as you have them
    mask = ((snap['age_days_kc_pre'] >= cooldown_kc_days) &
            ((snap['age_days_item_pre'] == 0.0) | (snap['age_days_item_pre'] >= cooldown_item_days)))
    cand = snap[mask].copy()
    if cand.empty:
        cand = snap.copy()

    # Predict
    p_hat = predict_lkt_p_hat(cand, params)
    cand = cand.assign(p_hat=p_hat)

    # Mastery gating
    pool = cand[cand['p_hat'] < mastery_p]
    # if pool.empty:
    #     pool = cand

    # ε-greedy exploration
    if random.random() < epsilon:
        # Prefer unseen items first; else least recently seen
        unseen = pool[pool['age_days_item_pre'] == 0.0]
        if not unseen.empty:
            return unseen.sample(1).iloc[0]
        # else pick among largest item-age
        pool = pool.sort_values(['age_days_item_pre'], ascending=False)
        # break any residual ties at random
        top_age = pool['age_days_item_pre'].iloc[0]
        top = pool[pool['age_days_item_pre'] == top_age]
        return top.sample(1).iloc[0]

    # Exploit near target
    pool = pool.assign(p_gap=(pool['p_hat'] - target_p).abs())

    # Tie-aware selection
    min_gap = pool['p_gap'].min()
    near = pool[pool['p_gap'] <= min_gap + tie_tol]

    # Prefer larger KC-age, then item-age; randomize remaining ties
    near = near.sort_values(['age_days_kc_pre', 'age_days_item_pre'], ascending=[False, False])
    top_kc_age = near['age_days_kc_pre'].iloc[0]
    near = near[near['age_days_kc_pre'] == top_kc_age]
    top_item_age = near['age_days_item_pre'].max()
    near = near[near['age_days_item_pre'] == top_item_age]
    return near.sample(1).iloc[0]


# -------------------------------------------------------------#

# - Neo4j Connection -#

driver = GraphDatabase.driver(
    "neo4j://127.0.0.1:7687",
    auth=("neo4j", NEO4J_PASSWORD)
)

# Comes out with big error if it can't connect, sorta makes a big box, easy to spot if connection fails
driver.verify_connectivity()


# -------------------------------------------------------------#

# - Retrieving Database Info -#


def debug_graph_structure(tx):
    """Check what node labels and relationships exist in the graph"""
    q = """
    CALL db.labels() YIELD label
    RETURN label
    ORDER BY label
    """
    return [dict(r) for r in tx.run(q)]


def debug_sample_nodes(tx):
    """Get sample nodes to see their structure"""
    q = """
    MATCH (n)
    RETURN labels(n) AS labels, 
           properties(n) AS props,
           id(n) AS neo_id
    LIMIT 20
    """
    return [dict(r) for r in tx.run(q)]


def debug_relationships(tx):
    """Check what relationships exist"""
    q = """
    MATCH (a)-[r]->(b)
    RETURN type(r) AS rel_type, 
           labels(a) AS from_labels, 
           labels(b) AS to_labels
    LIMIT 20
    """
    return [dict(r) for r in tx.run(q)]


# Add this before load_kc_catalog to debug:
with driver.session() as session:
    print("\n=== Node Labels ===")
    labels = session.execute_read(debug_graph_structure)
    for l in labels:
        print(l)

    print("\n=== Sample Nodes ===")
    nodes = session.execute_read(debug_sample_nodes)
    for n in nodes:
        print(n)

    print("\n=== Sample Relationships ===")
    rels = session.execute_read(debug_relationships)
    for r in rels:
        print(r)


def load_kc_catalog(driver) -> pd.DataFrame:
    with driver.session() as session:
        rows = session.execute_read(get_all_kc_subject)
    kc_catalog = pd.DataFrame(rows)
    if kc_catalog.empty:
        raise ValueError("KC catalog is empty.")
    kc_catalog['kc_id'] = kc_catalog['kc_id'].astype(int)
    kc_catalog['item_id'] = kc_catalog['item_id'].astype(int)

    return kc_catalog


def run_mvp_loop(
        user_id: int,
        state: OnlineState,
        item_catalog: pd.DataFrame,
        steps: int = 1,
        target_p: float = 0.70,
        mastery_p: float = 0.95,
        cooldown_kc_days: float = 0.1,
        cooldown_item_days: float = 0.01,
        sample_by_p: bool = False,
        seed: Optional[int] = 42
):
    rng = np.random.default_rng(seed)
    now = datetime.now(timezone.utc).replace(microsecond=0)

    for step in range(steps):
        suggestion = select_next_item_lkt(
            user_id=user_id,
            now=now,
            state=state,
            item_catalog=item_catalog,
            target_p=target_p,
            mastery_p=mastery_p,
            cooldown_kc_days=cooldown_kc_days,
            cooldown_item_days=cooldown_item_days
        )
        if suggestion is None:
            print("No item available. Catalog empty?")
            break

        # Simulate outcome
        if sample_by_p:
            p = float(suggestion.p_hat)
            outcome = int(rng.random() < p)
        else:
            outcome = int(rng.integers(0, 2))  # pure random

        # Record outcome and update state
        record_outcome_and_update(
            user_id=user_id,
            kc_id=int(suggestion.kc_id),
            item_id=int(suggestion.item_id),
            outcome=int(outcome),
            state=state,
            ts=now,
        )

        # Console trace
        print(
            f"[{now.isoformat()}] Ask item {suggestion.item_id} ({suggestion.item_name}) "
            f"KC {suggestion.kc_id} ({suggestion.kc_name}) | "
            f"p_hat={suggestion.p_hat:.2f} | "
            f"ageKC={suggestion.age_days_kc_pre:.2f}d ageItem={suggestion.age_days_item_pre:.2f}d | "
            f"outcome={outcome} | "
            f"user={user_id}"
        )

        # Advance simulated time a bit (5 min to ~1 day)
        dt_minutes = int(rng.integers(5, 24 * 60))
        now = now + pd.Timedelta(minutes=dt_minutes)


# -------------------------------------------------------------#

user_states = {}

user_id_1 = 1
user_id_2 = 2
if user_id_1 not in user_states:
    user_states[user_id_1] = OnlineState()

if user_id_2 not in user_states:
    user_states[user_id_2] = OnlineState()
kc_catalog = load_kc_catalog(driver)

state = OnlineState()

run_mvp_loop(
    user_id=user_id_1,
    state=user_states[user_id_1],
    item_catalog=kc_catalog,
    steps=50,
    target_p=0.6,
    mastery_p=0.95,
    cooldown_kc_days=0.1,
    cooldown_item_days=0.01,
    sample_by_p=False,
)

run_mvp_loop(
    user_id=user_id_2,
    state=user_states[user_id_2],
    item_catalog=kc_catalog,
    steps=50,
    target_p=0.6,
    mastery_p=0.95,
    cooldown_kc_days=0.1,
    cooldown_item_days=0.01,
    sample_by_p=False,
)

run_mvp_loop(
    user_id=user_id_1,
    state=user_states[user_id_1],
    item_catalog=kc_catalog,
    steps=50,
    target_p=0.7,
    mastery_p=0.95,
    cooldown_kc_days=0.1,
    cooldown_item_days=0.01,
    sample_by_p=False,
)