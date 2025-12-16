from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer
from evaluate import load
import numpy as np
import torch
import logging
import transformers

logging.basicConfig(level=logging.INFO)
transformers.utils.logging.set_verbosity_info()

accuracy = load("accuracy")
f1 = load("f1")
mean_absolute_error = load("mae")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    
    acc_result = accuracy.compute(predictions=preds, references=labels)
    macro_f1_result = f1.compute(predictions=preds, references=labels, average="macro")
    weighted_f1_result = f1.compute(predictions=preds, references=labels, average="weighted")
    mae_result = mean_absolute_error.compute(predictions=preds, references=labels)

    return {
        "accuracy": acc_result["accuracy"],
        "macro_f1": macro_f1_result["f1"],
        "weighted_f1": weighted_f1_result["f1"],
        "MAE": mae_result["mae"],
    }

def main():
    model_name = "bert-base-uncased"
    num_classes = 5
    classification_type = "single_label_classification"
    data_path = "../final_labeled_questions.json"

    # Read and split data
    df = pd.read_json(data_path, lines=False)
    X_train, X_test, y_train, y_test = train_test_split(
        df["question"],
        df["multiclass"],
        test_size=0.2,
        random_state=42,
        stratify=df["multiclass"]
    )

    # Shift labels to 0-indexed
    y_train = y_train - 1  # 1→0, 2→1, 3→2, 4→3, 5→4
    y_test = y_test - 1

    # Transform to HF datasets
    train_df = pd.DataFrame({"text": X_train, "labels": y_train.astype(int)})
    test_df  = pd.DataFrame({"text": X_test,  "labels": y_test.astype(int)})
    hf_train = Dataset.from_pandas(train_df)
    hf_test  = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256 # 512 ~ 384 words, 256 tokens ~ 192 words
        )

    hf_train = hf_train.map(preprocess, batched=True)
    hf_test  = hf_test.map(preprocess,  batched=True)

    hf_train.set_format("torch")
    hf_test.set_format("torch")

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,         
        problem_type=classification_type
    )

    # TODO: Make this a config
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        report_to=["wandb"],
        run_name="bert-multiclass-finetuning-7-epochs",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=7,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train,
        eval_dataset=hf_test,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    final_eval_metrics = trainer.evaluate()
    print(f"Final Eval Metrics: {final_eval_metrics}")

if __name__ == '__main__':
    main()


"""
Final Eval Metrics: {'eval_loss': 0.6593396067619324, 'eval_accuracy': 0.78, 'eval_macro_f1': 0.4893785310734463, 'eval_weighted_f1': 0.7535153349475384, 'eval_MAE': 0.3, 'eval_runtime': 3.3763, 'eval_samples_per_second': 59.236, 'eval_steps_per_second': 3.85, 'epoch': 3.0}
Final Eval Metrics: {'eval_loss': 0.6123411059379578, 'eval_accuracy': 0.805, 'eval_macro_f1': 0.5284412317444607, 'eval_weighted_f1': 0.7851879213477629, 'eval_MAE': 0.245, 'eval_runtime': 3.2804, 'eval_samples_per_second': 60.968, 'eval_steps_per_second': 3.963, 'epoch': 5.0}
Final Eval Metrics: {'eval_loss': 0.6385963559150696, 'eval_accuracy': 0.795, 'eval_macro_f1': 0.5176408341114224, 'eval_weighted_f1': 0.7751009959539371, 'eval_MAE': 0.25, 'eval_runtime': 3.2094, 'eval_samples_per_second': 62.317, 'eval_steps_per_second': 4.051, 'epoch': 7.0}
"""