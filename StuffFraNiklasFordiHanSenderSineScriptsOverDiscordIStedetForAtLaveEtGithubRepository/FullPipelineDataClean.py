import argparse
import json
from pathlib import Path
from typing import Dict, Any

# Import your existing modules (must be in the same directory or on PYTHONPATH)
import clean_dataset
import gather_matched_assets


def combine_labels_for_subject(subject_path: Path) -> Path:
    """
    Combine all JSON label files under subject_path / 'labels'
    into a single combined_labels.json in subject_path.
    Returns the path to the combined_labels.json (or None if nothing done).
    """
    labels_dir = subject_path / "labels"
    if not labels_dir.is_dir():
        print(f"[WARN] {subject_path.name}: no 'labels' directory, skipping JsonCombiner step.")
        return None

    json_files = sorted(labels_dir.glob("*.json"))
    if not json_files:
        print(f"[WARN] {subject_path.name}: no JSON files in 'labels', skipping.")
        return None

    combined: Dict[str, Dict[str, Any]] = {}

    for path in json_files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to read {path}: {e}")
            continue

        if not isinstance(data, list):
            print(f"[WARN] {path} does not contain a list of entries, skipping.")
            continue

        for entry in data:
            dt = entry.get("datetime")
            if not dt:
                continue

            if dt not in combined:
                combined[dt] = {"datetime": dt}

            # Merge attention/emotion if present
            if "attention" in entry and entry["attention"]:
                combined[dt]["attention"] = entry["attention"]
            if "emotion" in entry and entry["emotion"]:
                combined[dt]["emotion"] = entry["emotion"]

    if not combined:
        print(f"[WARN] {subject_path.name}: no valid entries with 'datetime' found.")
        return None

    merged_list = list(combined.values())
    merged_list.sort(key=lambda x: x["datetime"])

    out_path = subject_path / "combined_labels.json"
    with out_path.open("w", encoding="utf-8") as out_f:
        json.dump(merged_list, out_f, indent=4, ensure_ascii=False)

    print(f"[JsonCombiner] {subject_path.name}: merged {len(json_files)} files"
          f" → {len(merged_list)} unique entries.")
    print(f"               Saved: {out_path}")
    return out_path


def run_pipeline(
    experiment_root: Path,
    clean_root: Path,
    matched_root: Path,
    copy_images_in_clean: bool = True,
    dry_run_matched: bool = False,
) -> None:
    experiment_root = experiment_root.resolve()
    clean_root = clean_root.resolve()
    matched_root = matched_root.resolve()

    if not experiment_root.exists():
        raise SystemExit(f"[ERROR] Experiment root not found: {experiment_root}")

    print(f"[INFO] Experiment root: {experiment_root}")
    print(f"[INFO] Clean root     : {clean_root}")
    print(f"[INFO] Matched root   : {matched_root}")

    # ------------------------------------------------------------------
    # 1) Combine labeler JSONs into combined_labels.json per subject
    # ------------------------------------------------------------------
    print("\n[STEP 1] Combining label JSONs per subject (JsonCombiner logic)...")
    subjects = sorted(
        p for p in experiment_root.iterdir()
        if p.is_dir() and p.name.startswith("subject_")
    )

    if not subjects:
        print("[WARN] No subject_* folders found under experiment root.")
    else:
        for subj in subjects:
            combine_labels_for_subject(subj)

    # ------------------------------------------------------------------
    # 2) Run clean_dataset.clean_dataset on the experiment root
    #    This will filter by attention + trim metadata into clean_root
    # ------------------------------------------------------------------
    print("\n[STEP 2] Running clean_dataset.clean_dataset...")
    clean_root.mkdir(parents=True, exist_ok=True)
    # Uses your existing function from clean_dataset.py
    clean_dataset.clean_dataset(
        in_root=experiment_root,
        out_root=clean_root,
        copy_images=copy_images_in_clean,
    )

    # ------------------------------------------------------------------
    # 3) Run gather_matched_assets.process_one_subject on cleaned data
    #    to create 'matched-only' copies in matched_root
    # ------------------------------------------------------------------
    print("\n[STEP 3] Running gather_matched_assets.process_one_subject on cleaned data...")
    matched_root.mkdir(parents=True, exist_ok=True)

    # Extract experiment number from the root folder name (e.g. "Experiment02" → "2")
    exp_name = experiment_root.name
    exp_num = ''.join(ch for ch in exp_name if ch.isdigit()) or "0"

    cleaned_subjects = sorted(p for p in clean_root.iterdir() if p.is_dir())
    if not cleaned_subjects:
        print("[WARN] No subject folders found in clean_root; nothing to match.")
        return

    for subj in cleaned_subjects:
        labels_json = subj / "combined_labels.json"
        if not labels_json.exists():
            print(f"[WARN] Skipping {subj.name}: no combined_labels.json in cleaned data.")
            continue

        # Extract subject number (e.g., "subject_03" -> "3")
        subj_num = ''.join(ch for ch in subj.name if ch.isdigit()) or "0"
        new_name = f"Exp{exp_num}Subject{subj_num}"

        subj_out = matched_root / new_name

        gather_matched_assets.process_one_subject(
            subject_dir=subj,
            labels_json=labels_json,
            out_dir=subj_out,
            dry_run=dry_run_matched,
        )

    print("\n[INFO] Pipeline finished.")
    print(f"[INFO] Cleaned dataset is in : {clean_root}")
    print(f"[INFO] Matched-only dataset : {matched_root}")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run full pipeline:\n"
            "1) Combine labelers into combined_labels.json per subject\n"
            "2) Clean dataset (filter by attention + prune metadata)\n"
            "3) Gather only matched images+metadata referenced in combined_labels.json"
        )
    )
    ap.add_argument(
        "--experiment_root",
        type=Path,
        required=True,
        help="Path to Experiment folder (e.g. Experiment01) containing subject_01, subject_02, ...",
    )
    ap.add_argument(
        "--clean_root",
        type=Path,
        help="Output root for cleaned dataset (default: <experiment_root>_clean)",
    )
    ap.add_argument(
        "--matched_root",
        type=Path,
        help="Output root for matched-only dataset (default: <experiment_root>_matched)",
    )
    ap.add_argument(
        "--no_images_in_clean",
        action="store_true",
        help="Do not copy images in the clean_dataset step (only labels + metadata).",
    )
    ap.add_argument(
        "--dry_run_matched",
        action="store_true",
        help="Dry run for matched-assets step (report only; do not copy files).",
    )

    args = ap.parse_args()

    experiment_root: Path = args.experiment_root
    clean_root: Path = args.clean_root or experiment_root.with_name(experiment_root.name + "_clean")
    matched_root: Path = args.matched_root or experiment_root.with_name(experiment_root.name + "_matched")

    run_pipeline(
        experiment_root=experiment_root,
        clean_root=clean_root,
        matched_root=matched_root,
        copy_images_in_clean=not args.no_images_in_clean,
        dry_run_matched=args.dry_run_matched,
    )


if __name__ == "__main__":
    main()
