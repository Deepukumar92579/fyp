# import details
# !pip install seqeval
# !pip install evaluate
# !pip install datasets


import json
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset, DatasetDict
import evaluate

# Load dataset from JSON file
file_path = "./sanskrit_ner_bio.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)


# Convert to Hugging Face Dataset format
def process_data(data):
    all_words = []
    all_labels = []
    unique_labels = set()

    for item in data:
        all_words.append(item["tokens"])
        all_labels.append(item["ner_tags"])
        unique_labels.update(item["ner_tags"])

    return all_words, all_labels, list(unique_labels)


words, labels, unique_labels = process_data(data)
label_to_id = {label: i for i, label in enumerate(unique_labels)}
id_to_label = {i: label for label, i in label_to_id.items()}


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=128,
        is_split_into_words=True,
    )

    labels = []
    for i, word_list in enumerate(examples["tokens"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        ner_tags = examples["ner_tags"][i]

        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id.get(ner_tags[word_idx], -100))
            else:
                label_ids.append(label_to_id.get(ner_tags[word_idx], -100))
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Create Hugging Face dataset
hf_dataset = Dataset.from_dict({"tokens": words, "ner_tags": labels})
train_test_split = hf_dataset.train_test_split(test_size=0.2)
datasets = DatasetDict(
    {"train": train_test_split["train"], "test": train_test_split["test"]}
)

model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


# Tokenize dataset
tokenized_datasets = datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=datasets["train"].column_names,
    load_from_cache_file=False,
)

# load model
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(unique_labels), ignore_mismatched_sizes=True
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
)


def compute_metrics(p):
    metric = evaluate.load("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[id_to_label[l] for l in label if l != -100] for label in labels]
    pred_labels = [
        [id_to_label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    return metric.compute(predictions=pred_labels, references=true_labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

# Evaluate model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Save model
model.save_pretrained("./fine_tuned_xlm_roberta")
tokenizer.save_pretrained("./fine_tuned_xlm_roberta")

print("Label to ID mapping:")
for label, idx in label_to_id.items():
    print(f"{label}: {idx}")

print("\nID to Label mapping:")
for idx, label in id_to_label.items():
    print(f"{idx}: {label}")