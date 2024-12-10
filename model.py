# Ensure required packages are installed
# !pip install transformers datasets openpyxl

# Import necessary libraries
import pandas as pd
from datasets import Dataset
from transformers import T5ForConditionalGeneration, ByT5Tokenizer, Trainer, TrainingArguments

# Load your Excel file into a pandas DataFrame
file_path = './sanskrit.xlsx'  # Replace with your actual file name
df = pd.read_excel(file_path)

# Ensure the columns are named correctly, based on your dataset
print(df.head())

# Convert DataFrame to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# Load the ByT5 tokenizer and T5 model
model_name = "google/byt5-small"
tokenizer = ByT5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Preprocess function to tokenize the dataset
def preprocess_function(examples):
    inputs = [ex for ex in examples['Unseg']]
    targets = [ex for ex in examples['Seg']]

    # Tokenize inputs and targets
    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
    labels = tokenizer(targets, max_length=512, padding="max_length", truncation=True)

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Apply preprocessing to the dataset
encoded_dataset = dataset.map(preprocess_function, batched=True)

# Split into training and validation datasets (optional, but recommended)
train_dataset = encoded_dataset.train_test_split(test_size=0.1)['train']
eval_dataset = encoded_dataset.train_test_split(test_size=0.1)['test']


training_args = TrainingArguments(
    output_dir="./results",          # output directory
    num_train_epochs=3,              # number of training epochs
    # per_device_train_batch_size=8,   # batch size for training
    # per_device_eval_batch_size=8,    # batch size for evaluation
    per_device_train_batch_size=2,   # Reduce batch size
    gradient_accumulation_steps=2,   # Accumulate gradients
    per_device_eval_batch_size=2,    
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,                # log every 10 steps
    evaluation_strategy="epoch",     # evaluate every epoch
    save_strategy="epoch",           # save model every epoch
    report_to="none",                # Disable WandB logging
)


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()