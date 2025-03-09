import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np

# Load the trained model and tokenizer
model_path = "./fine_tuned_xlm_roberta" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Mapping from label IDs to labels (verify from the output during model training)
id_to_label = {
    2: "O",
    1: "B-PER",
    0: "B-WEAPON",
    4: "B_LOC",
    3: "B_ORG"
}

label_to_id = {v: k for k, v in id_to_label.items()} #no need to define here, get it from the model

print(id_to_label)

def read_sanskrit_sentences_from_xlsx(file_path, column_name, num_samples=None):
    df = pd.read_excel(file_path)
    sentences = df[column_name].dropna().tolist()
    return sentences[:num_samples] if num_samples else sentences

def predict_with_adjusted_threshold(logits, per_threshold=3.5, loc_threshold=3.5):
    predictions = []
    for token_logits in logits:
        # Check for B-PER (index 1)     
        per_condition = (token_logits[label_to_id['O']] - token_logits[label_to_id['B-PER']] < per_threshold and 
                        token_logits[label_to_id['B-PER']] > 0 and
                        token_logits[label_to_id['B-PER']] > token_logits[label_to_id['B-LOC']] and 
                        token_logits[label_to_id['B-PER']] > token_logits[label_to_id['B-ORG']] and 
                        token_logits[label_to_id['B-PER']] > token_logits[label_to_id['B-WEAPON']])
         
        # Check for B-LOC (index 3)
        loc_condition = (token_logits[label_to_id['O']] - token_logits[label_to_id['B-LOC']] < loc_threshold and 
                        token_logits[label_to_id['B-LOC']] > 0 and
                        token_logits[label_to_id['B-LOC']] > token_logits[label_to_id['B-PER']] and 
                        token_logits[label_to_id['B-LOC']] > token_logits[label_to_id['B-ORG']] and 
                        token_logits[label_to_id['B-LOC']] > token_logits[label_to_id['B-WEAPON']])

        
        if per_condition:
            predictions.append(1)  # B-PER
        elif loc_condition:
            predictions.append(3)  # B-LOC
        else:
            predictions.append(torch.argmax(token_logits).item())
    
    return predictions

# Function to predict NER tags for given Sanskrit sentences
def predict_ner(sentences, use_threshold=True, threshold_value=3.5, verbose=False):
    model.eval() 
    
    results = []

    for sentence in sentences:
        # Tokenize sentence with word alignment
        inputs = tokenizer(sentence.split(), return_tensors="pt", is_split_into_words=True, padding=True, truncation=True)

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        word_ids = inputs.word_ids(0)
        
        if verbose:
            print(f"Total tokens: {len(tokens)}")
            print(f"Token to word mapping: {list(zip(tokens[:20], word_ids[:20]))}")  # First 20 tokens

        with torch.no_grad():
            outputs = model(**inputs)

        # Get the raw logits
        logits = outputs.logits[0]
        
        if verbose:
            print(f"Logits shape: {logits.shape}")

            # Print logits for all tokens
            for idx, token in enumerate(tokens):
                print(f"Token: {token}, Logits: {logits[idx].tolist()}")

        if use_threshold:
            predictions = predict_with_adjusted_threshold(logits, per_threshold=threshold_value, loc_threshold = threshold_value)
        else:
            predictions = torch.argmax(logits, dim=1).tolist()
            
        if verbose:
            # Print unique prediction labels
            unique_preds = set(predictions)
            print(f"Unique prediction indices: {unique_preds}")

        # Align predictions to words
        word_predictions = {}
        for i, (word_id, pred) in enumerate(zip(word_ids, predictions)):
            if word_id is not None:  # Skip special tokens
                # If this is the first token for this word or has higher confidence for a named entity, use it
                if word_id not in word_predictions or (pred != 0 and word_predictions.get(word_id) == 0):
                    word_predictions[word_id] = pred

        # Convert to final labels
        aligned_labels = [id_to_label.get(word_predictions.get(i, 0), "O") for i in range(len(sentence.split()))]

        print("\nSentence:", sentence)
        print("NER Predictions:")
        
        # Collect word-label pairs for this sentence
        sentence_results = []
        for word, label in zip(sentence.split(), aligned_labels):
            print(f"{word}: {label}")
            sentence_results.append((word, label))
            
        results.append(sentence_results)
        
    return results

if __name__ == "__main__":
    xlsx_file_path = "/content/Cleaned_Gita.xlsx" 
    column_name = "Seg" 
    num_samples = 100 
    
    # Read the Sanskrit sentences from the Excel file
    sanskrit_sentences = read_sanskrit_sentences_from_xlsx(xlsx_file_path, column_name, num_samples)
    
    print(f"Loaded {len(sanskrit_sentences)} sentences from {xlsx_file_path}")
    
    # Run prediction with different settings
    print("\n Standard Prediction (No Adjustments) ")
    standard_results = predict_ner(sanskrit_sentences, use_threshold=False)
    
    print("\n Threshold-Adjusted Prediction ")
    threshold_results = predict_ner(sanskrit_sentences, use_threshold=True, threshold_value=3.5)
    
    # Save results to Excel
    results_df = pd.DataFrame(columns=["Sentence", "Word", "Standard_Label", "Threshold_Label"])
    
    for i, sentence in enumerate(sanskrit_sentences):
        words = sentence.split()
        for j, word in enumerate(words):
            results_df = results_df._append({
                "Sentence": sentence,
                "Word": word,
                "Standard_Label": standard_results[i][j][1],
                "Threshold_Label": threshold_results[i][j][1]
            }, ignore_index=True)
    
    results_df.to_excel("ner_prediction_results.xlsx", index=False)
    print("Results saved to ner_prediction_results.xlsx")
