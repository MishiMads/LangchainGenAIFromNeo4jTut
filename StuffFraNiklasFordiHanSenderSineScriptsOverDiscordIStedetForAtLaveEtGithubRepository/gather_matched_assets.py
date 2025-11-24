import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

IMAGE_EXTS = [".jpg", ".jpeg", ".png"]


def normalize_datetime(dt: str) -> str:
    return dt.replace(":", "_").strip()


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def find_images_by_stem(subject_dir: Path, stem: str) -> List[Path]:
    """Prefer exact stem match; fall back to contains-stem match."""
    matches_exact: List[Path] = []
    matches_contains: List[Path] = []

    for ext in IMAGE_EXTS:
        matches_exact += list(subject_dir.rglob(f"{stem}{ext}"))
        matches_contains += list(subject_dir.rglob(f"*{stem}*{ext}"))

    # Deduplicate while preserving "exact first"
    seen = set()
    ordered: List[Path] = []
    for lst in (matches_exact, matches_contains):
        for p in sorted(lst):
            if p not in seen:
                ordered.append(p)
                seen.add(p)
    return ordered


def find_metadata_for_image(subject_dir: Path, image_path: Path) -> Optional[Path]:
    stem = image_path.stem
    same_dir = image_path.with_suffix(".json")
    if same_dir.exists():
        return same_dir
    # fallback: anywhere under subject_dir
    alts = sorted(subject_dir.rglob(f"{stem}.json"))
    return alts[0] if alts else None


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path):
    ensure_dir(dst.parent)
    dst.write_bytes(src.read_bytes())


def process_one_subject(subject_dir: Path, labels_json: Path, out_dir: Path, dry_run: bool) -> None:
    subject_dir = subject_dir.resolve()
    labels_json = labels_json.resolve()
    out_dir = out_dir.resolve()

    if not subject_dir.exists():
        print(f"[ERROR] subject_dir not found: {subject_dir}", file=sys.stderr)
        return
    if not labels_json.exists():
        print(f"[ERROR] labels_json not found: {labels_json}", file=sys.stderr)
        return

    print(f"[INFO] Processing subject: {subject_dir.name}")
    labels = read_json(labels_json)
    if not isinstance(labels, list):
        print(f"[ERROR] combined_labels.json must be a list of objects in {labels_json}", file=sys.stderr)
        return

    images_out = out_dir / "images"
    metadata_out = out_dir / "metadata"
    ensure_dir(images_out)
    ensure_dir(metadata_out)

    matched = 0
    skipped = 0

    for item in labels:
        stem = None
        if "filename" in item and item["filename"]:
            stem = Path(item["filename"]).stem
        elif "datetime" in item and item["datetime"]:
            stem = normalize_datetime(str(item["datetime"]))
        else:
            skipped += 1
            continue

        img_candidates = find_images_by_stem(subject_dir, stem)
        if not img_candidates:
            skipped += 1
            continue
        img = img_candidates[0]

        meta = find_metadata_for_image(subject_dir, img)
        if not meta or not meta.exists():
            skipped += 1
            continue

        if not dry_run:
            copy_file(img, images_out / img.name)
            copy_file(meta, metadata_out / meta.name)
        matched += 1

    if not dry_run:
        copy_file(labels_json, out_dir / "combined_labels.json")

    print(f"[DONE] {subject_dir.name}: Matched & copied pairs: {matched} | Skipped: {skipped}")
    print(f"        Out: {out_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Create folders with ONLY pictures + metadata referenced in combined_labels.json"
    )

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--subject_dir", type=Path, help="Single subject folder to search (e.g., Exp2Subject1)")
    mode.add_argument("--in_root", type=Path, help="Root folder containing many subject folders")

    ap.add_argument("--labels_json", type=Path, help="combined_labels.json path (single-subject mode only)")
    ap.add_argument("--out_dir", type=Path, help="Destination folder (single-subject mode only)")
    ap.add_argument("--out_root", type=Path, help="Destination root for all subjects (whole-experiment mode)")
    ap.add_argument("--dry_run", action="store_true", help="Report only; do not copy files")

    args = ap.parse_args()

    # --- Single-subject mode ---
    if args.subject_dir is not None:
        if args.labels_json is None or args.out_dir is None:
            print("[ERROR] --subject_dir requires BOTH --labels_json and --out_dir", file=sys.stderr)
            sys.exit(1)
        process_one_subject(args.subject_dir, args.labels_json, args.out_dir, args.dry_run)
        return

    # --- Whole-experiment mode ---
    in_root: Path = args.in_root
    out_root: Optional[Path] = args.out_root

    if out_root is None:
        print("[ERROR] --in_root requires --out_root", file=sys.stderr)
        sys.exit(1)

    in_root = in_root.resolve()
    out_root = out_root.resolve()

    if not in_root.exists():
        print(f"[ERROR] in_root not found: {in_root}", file=sys.stderr)
        sys.exit(1)

    ensure_dir(out_root)
    print(f"[INFO] Whole-experiment mode")
    print(f"[INFO] in_root  = {in_root}")
    print(f"[INFO] out_root = {out_root}")

    # Extract experiment number from folder name like "Experiment02"
    exp_name = in_root.name
    exp_num = ''.join(ch for ch in exp_name if ch.isdigit()) or "0"

    subjects = sorted(p for p in in_root.iterdir() if p.is_dir())
    if not subjects:
        print("[WARN] No subject directories found under in_root.")
        sys.exit(0)

    for subj in subjects:
        labels = subj / "combined_labels.json"
        if not labels.exists():
            print(f"[WARN] Skipping {subj.name}: no combined_labels.json")
            continue

        # Extract subject number (e.g., "subject_03" -> "3")
        subj_num = ''.join(ch for ch in subj.name if ch.isdigit()) or "0"

        # Construct new folder name
        new_name = f"Exp{exp_num}Subject{subj_num}"
        subj_out = out_root / new_name

        process_one_subject(subj, labels, subj_out, args.dry_run)

    print("[INFO] Done processing all subjects.")


if __name__ == "__main__":
    main()
