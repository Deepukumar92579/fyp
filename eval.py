import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu # type: ignore

model_name = "/kaggle/working/saved_model2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

input_file_path = "/kaggle/input/test-pat/test_patanjali(1).xlsx"
df = pd.read_excel(input_file_path)

input_texts = df['Unseg'].tolist()
references = df['Seg'].tolist()

total_bleu_score = 0
num_samples = len(input_texts)

for idx, test_input in enumerate(input_texts):
    input_ids = tokenizer(test_input, return_tensors="pt", max_length=128, padding="max_length", truncation=True).input_ids.to(device)

    predicted_ids = model.generate(
        input_ids,
        max_length=128,
        num_beams=4,
        repetition_penalty=2.0,
        early_stopping=True
    )

    output = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

    reference = [references[idx]]

    reference_tokens = [ref.split() for ref in reference]
    generated_tokens = output.split()
    bleu_score = sentence_bleu(reference_tokens, generated_tokens)
    total_bleu_score += bleu_score

average_bleu_score = total_bleu_score / num_samples

print(f"Average BLEU Score for the Dataset: {average_bleu_score:.8f}")