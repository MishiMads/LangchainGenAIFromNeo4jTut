
from pathlib import Path
import json, shutil, re
from typing import Dict, Any, Iterable, Optional

def ts_to_key(ts: str) -> str:
    """Convert '10:40:43:015111' -> '10_40_43_015111'"""
    return ts.replace(":", "_")

def key_to_ts(key: str) -> str:
    return key.replace("_", ":")

def load_attention_keys(labels_path: Path) -> Dict[str, Dict[str, Any]]:
    """Return a dict mapping ts_key -> label_row for rows that have 'attention'."""
    with labels_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    keep = {}
    for r in rows:
        if "attention" in r and r["attention"] not in (None, "", "null"):
            k = ts_to_key(str(r["datetime"]))
            keep[k] = r
    return keep

def prune_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    person = meta.get("person", {})
    cleaned = {"person": {}}

    def keep_bbox(d):
        bb = d.get("bounding_box")
        return {"bounding_box": bb} if isinstance(bb, dict) else None

    def keep_pose(d):
        out = {}
        if isinstance(d.get("headpose"), dict) and isinstance(d["headpose"].get("pose"), dict):
            p = d["headpose"]["pose"]
            out["headpose"] = {"pose": {k: float(p.get(k, 0.0)) for k in ("yaw","pitch","roll")}}
        if isinstance(d.get("pose"), dict):
            out["pose"] = d["pose"]
        if "body_pose" in d:  # list of landmarks is fine to keep
            out["body_pose"] = d["body_pose"]
        return out

    # Face
    face = person.get("face", {})
    face_keep = {}
    if isinstance(face, dict):
        bb = keep_bbox(face)
        if bb: face_keep.update(bb)
        face_keep.update(keep_pose(face))
    if face_keep:
        cleaned["person"]["face"] = face_keep

    # Body (+ hand_pose nested inside body)
    body = person.get("body", {})
    body_keep = {}
    if isinstance(body, dict):
        bb = keep_bbox(body)
        if bb: body_keep.update(bb)
        body_keep.update(keep_pose(body))

        # NEW: preserve body.hand_pose.left_hand/right_hand if present
        hp = body.get("hand_pose")
        if isinstance(hp, dict):
            out_hp = {}
            for side in ("left_hand", "right_hand"):
                if isinstance(hp.get(side), list) and hp[side]:
                    out_hp[side] = hp[side]
            if out_hp:
                body_keep["hand_pose"] = out_hp

    if body_keep:
        cleaned["person"]["body"] = body_keep

    # (Optionally keep any legacy hand schemas too)
    for name in ("hands", "hand_left", "hand_right"):
        if name in person and isinstance(person[name], (dict, list)):
            cleaned["person"][name] = person[name]

    return cleaned

def clean_subject(subject_dir: Path, out_root: Path, copy_images: bool=True, image_exts=(".png",".jpg",".jpeg")) -> None:
    subj_name = subject_dir.name
    labels_path = subject_dir / "combined_labels.json"
    images_dir = subject_dir / "images"
    meta_dir = subject_dir / "metadata"

    if not labels_path.exists():
        print(f"[WARN] No labels for {subj_name}, skipping.")
        return
    keep_map = load_attention_keys(labels_path)
    if not keep_map:
        print(f"[WARN] No attention labels in {subj_name}, skipping.")
        return

    out_dir = out_root / subj_name
    out_images = out_dir / "images"
    out_meta = out_dir / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    # write filtered labels
    filtered_rows = [keep_map[k] for k in sorted(keep_map.keys())]
    with (out_dir / "combined_labels.json").open("w", encoding="utf-8") as f:
        json.dump(filtered_rows, f, indent=2)

    # copy corresponding frames
    for ts_key in keep_map.keys():
        # image
        img_src = None
        for ext in image_exts:
            cand = images_dir / f"{ts_key}{ext}"
            if cand.exists():
                img_src = cand
                break
        if img_src and copy_images:
            shutil.copy2(img_src, out_images / img_src.name)
        elif copy_images:
            print(f"[WARN] image not found for {ts_key} in {subj_name}")

        # metadata
        meta_src = meta_dir / f"{ts_key}.json"
        if meta_src.exists():
            try:
                with meta_src.open("r", encoding="utf-8") as f:
                    j = json.load(f)
                cleaned = prune_metadata(j)
                with (out_meta / meta_src.name).open("w", encoding="utf-8") as fo:
                    json.dump(cleaned, fo, indent=2)
            except Exception as e:
                print(f"[ERROR] pruning {meta_src}: {e}")
        else:
            print(f"[WARN] metadata not found for {ts_key} in {subj_name}")

def clean_dataset(in_root: Path, out_root: Path, copy_images: bool=True):
    out_root.mkdir(parents=True, exist_ok=True)
    subjects = sorted([p for p in in_root.iterdir() if p.is_dir()])
    for subj in subjects:
        if (subj / "combined_labels.json").exists():
            clean_subject(subj, out_root, copy_images=copy_images)
    print(f"Done. Cleaned data at: {out_root}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Clean engagement dataset (filter by attention, trim metadata).")
    ap.add_argument("--in_root", type=Path, required=True, help="Folder containing Exp*Subject* folders.")
    ap.add_argument("--out_root", type=Path, required=True, help="Destination for cleaned dataset.")
    ap.add_argument("--no_images", action="store_true", help="Do not copy images (keep only labels + metadata).")
    args = ap.parse_args()
    clean_dataset(args.in_root, args.out_root, copy_images=not args.no_images)
