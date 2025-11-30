import json
import pandas as pd

# === CONFIG ===
json_path = "Nicholas_questions.json"         # your input JSON file
excel_path = "labels.xlsx"           # your Excel file
output_path = "Nicholas_questions_labeled_final.json"  # output JSON file

# === LOAD DATA ===
with open(json_path, "r") as f:
    data = json.load(f)

labels = pd.read_excel(excel_path)

# # Sanity check
# if len(data) != len(labels):
#     raise ValueError(f"JSON has {len(data)} items but Excel has {len(labels)} rows — they must match in order!")

# === ADD LABELS TO QUESTIONS ===
for i, item in enumerate(data[:500]):
    item["binary"] = int(labels.iloc[i]["Binary"])
    item["multiclass"] = int(labels.iloc[i]["Multi-class"])

# === SAVE UPDATED JSON ===
with open(output_path, "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ Labeled JSON saved to {output_path}")
