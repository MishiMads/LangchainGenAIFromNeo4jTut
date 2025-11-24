import json
import glob
import os
from collections import defaultdict

# Root directory containing all subjects
root_dir = "EngagementTemp"

# Loop through each subject folder
for subject_path in glob.glob(os.path.join(root_dir, "subject_*")):
    label_dir = os.path.join(subject_path, "labels")
    if not os.path.isdir(label_dir):
        continue  # skip if no labels folder

    # Find all JSON label files inside the labels folder
    json_files = glob.glob(os.path.join(label_dir, "*.json"))
    if not json_files:
        print(f"No label files found in {label_dir}")
        continue

    combined = {}
    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            dt = entry.get("datetime")
            if not dt:
                continue

            if dt not in combined:
                combined[dt] = {"datetime": dt}

            # Merge attention/emotion values if present and not already filled
            if "attention" in entry and entry["attention"]:
                combined[dt]["attention"] = entry["attention"]
            if "emotion" in entry and entry["emotion"]:
                combined[dt]["emotion"] = entry["emotion"]

    # Convert to list and sort by datetime
    merged_list = list(combined.values())
    merged_list.sort(key=lambda x: x["datetime"])

    # Output file in the subject's main folder
    output_path = os.path.join(subject_path, "combined_labels.json")
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(merged_list, out, indent=4, ensure_ascii=False)

    print(f"Merged {len(json_files)} files → {len(merged_list)} unique entries for {os.path.basename(subject_path)}")
    print(f"Saved as {output_path}")
