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
precision = load("precision")
recall = load("recall")
f1 = load("f1")
roc_auc = load("roc_auc")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)

    # Softmax for probabilities (logits shape is [batch, 2] for binary)
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    
    acc_result = accuracy.compute(predictions=preds, references=labels)
    prec_result = precision.compute(predictions=preds, references=labels, average="binary") 
    rec_result = recall.compute(predictions=preds, references=labels, average="binary")
    f1_result = f1.compute(predictions=preds, references=labels, average="binary")
    auc_result = roc_auc.compute(prediction_scores=probabilities[:, 1], references=labels)

    return {
        "accuracy": acc_result["accuracy"],
        "precision": prec_result["precision"],
        "recall": rec_result["recall"],
        "f1": f1_result["f1"],
        "roc_auc": auc_result["roc_auc"]
    }

def main():
    model_name = "bert-base-uncased"
    num_classes = 2
    classification_type = "single_label_classification"
    data_path = "../final_labeled_questions.json"

    # Read and split data
    df = pd.read_json(data_path, lines=False)
    X_train, X_test, y_train, y_test = train_test_split(
        df["question"],
        df["binary"],
        test_size=0.2,
        random_state=42,
        stratify=df["binary"]
    )

    # Transform to HF datasets
    train_df = pd.DataFrame({"text": X_train, "labels": y_train})
    test_df  = pd.DataFrame({"text": X_test,  "labels": y_test})
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
        run_name="bert-binary-finetuning-3-epochs",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
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
Final Eval Metrics: {'eval_loss': 0.3229988217353821, 'eval_accuracy': 0.875, 'eval_precision': 0.8627450980392157, 'eval_recall': 0.8888888888888888, 'eval_f1': 0.8756218905472637, 'eval_roc_auc': 0.9317931793179317, 'eval_runtime': 3.2365, 'eval_samples_per_second': 61.795, 'eval_steps_per_second': 4.017, 'epoch': 3.0}
Final Eval Metrics: {'eval_loss': 0.37455153465270996, 'eval_accuracy': 0.87, 'eval_precision': 0.8543689320388349, 'eval_recall': 0.8888888888888888, 'eval_f1': 0.8712871287128713, 'eval_roc_auc': 0.9260926092609261, 'eval_runtime': 3.2051, 'eval_samples_per_second': 62.401, 'eval_steps_per_second': 4.056, 'epoch': 5.0}
Final Eval Metrics: {'eval_loss': 0.3244902491569519, 'eval_accuracy': 0.88, 'eval_precision': 0.8947368421052632, 'eval_recall': 0.8585858585858586, 'eval_f1': 0.8762886597938144, 'eval_roc_auc': 0.9265926592659266, 'eval_runtime': 3.2599, 'eval_samples_per_second': 61.352, 'eval_steps_per_second': 3.988, 'epoch': 3.0}
"""