from datasets import load_dataset, Dataset
import random
import json
import math

# Set random seed for reproducibility
random.seed(42)

# Number of samples from each dataset
samples_per_dataset = [333, 333, 334]  # sum = 1000

# ---- 1. GSM8K ----
gsm8k = load_dataset("gsm8k", "main", split="train")
gsm8k_questions = [{"question": item["question"], "answer": item["answer"]} for item in gsm8k]
gsm8k_sample = random.sample(gsm8k_questions, samples_per_dataset[0]) # without replacement

# ---- 2. SQuAD v1.1 ----
squad = load_dataset("squad", split="train")
squad_questions = [
    {
        "question": item['context'] + '\n' + item["question"],
        "answer": item["answers"]["text"][0] if len(item["answers"]["text"]) > 0 else ""
    }
    for item in squad
]
squad_sample = random.sample(squad_questions, samples_per_dataset[1])

# ---- 3. MMLU ----
# Categories to sample from

categories = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 
              'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 
              'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 
              'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 
              'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 
              'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 
              'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 
              'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 
              'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 
              'sociology', 'us_foreign_policy', 'virology', 'world_religions']
mmlu_questions = []

num_categories = len(categories)
questions_per_category = samples_per_dataset[2] / num_categories # is about 5.6

for category in categories:
    dataset = load_dataset("cais/mmlu", category, split="test")
    len_dataset = len(dataset)
    print(f"Length of Category {category}: {len_dataset}")

    # Convert Dataset to list first
    dataset_list = list(dataset)
    
    # Sample from the list
    if len_dataset >= math.ceil(questions_per_category):
        mmlu_questions.extend(random.sample(dataset_list, math.ceil(questions_per_category)))
    else:
        mmlu_questions.extend(dataset_list)

print("MMLU questions ", len(mmlu_questions))
# Trim to exactly 334
mmlu_sample = random.sample(mmlu_questions, samples_per_dataset[2])

print(f"Collected {len(mmlu_sample)} MMLU questions from selected categories.")

# ---- Combine ----
combined = gsm8k_sample + squad_sample + mmlu_sample
random.shuffle(combined)

print("TOTAL DATASET LENGTH: ", len(combined))

# ---- Save to JSON ----
with open("combined_1000_questions.json", "w") as f:
    json.dump(combined, f, indent=2)

print("Combined dataset of 1000 questions saved as 'combined_1000_questions.json'.")
