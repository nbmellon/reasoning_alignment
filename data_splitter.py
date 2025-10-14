import json
import random

# Load the full dataset
with open("combined_1000_questions.json", "r") as f:
    all_questions = json.load(f)

# Shuffle to randomize
random.shuffle(all_questions)

# Split into two halves
nicholas_questions = all_questions[:500]
evan_questions = all_questions[500:]

# Save to separate JSON files
with open("Nicholas_questions.json", "w") as f:
    json.dump(nicholas_questions, f, indent=2)

with open("Evan_questions.json", "w") as f:
    json.dump(evan_questions, f, indent=2)

print("Saved Nicholas_questions.json and Evan_questions.json, each with 500 questions.")
