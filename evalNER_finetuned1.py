import torch
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

print(id_to_label)
# List of Sanskrit sentences to test
sanskrit_sentences = [
    "अयोध्यायां दशरथः राजा सत्यवादी धर्मात्मा।"
]

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
        loc_condition = (token_logits[0] - token_logits[3] < loc_threshold and 
                        token_logits[3] > 0 and
                        token_logits[3] > token_logits[1] and 
                        token_logits[3] > token_logits[2] and 
                        token_logits[3] > token_logits[4])
        
        if per_condition:
            predictions.append(1)  # B-PER
        elif loc_condition:
            predictions.append(3)  # B-LOC
        else:
            predictions.append(torch.argmax(token_logits).item())
    
    return predictions

def predict_ner(sentences, use_threshold=True, threshold_value=4.5):
    model.eval()

    for sentence in sentences:
        inputs = tokenizer(sentence.split(), return_tensors="pt", is_split_into_words=True, padding=True, truncation=True)

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        word_ids = inputs.word_ids(0)
        print(f"Token to word mapping: {list(zip(tokens[:20], word_ids[:20]))}")  # first 20 tokens

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0]
        print(f"Logits shape: {logits.shape}")

        for idx, token in enumerate(tokens):
            print(f"Token: {token}, Logits: {logits[idx].tolist()}")

        # Apply different prediction methods based on parameters
        if use_threshold:
            predictions = predict_with_adjusted_threshold(logits, per_threshold=4.0, loc_threshold=3.5)           
        else:
            # Standard prediction (argmax)
            predictions = torch.argmax(logits, dim=1).tolist()
            
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
        for word, label in zip(sentence.split(), aligned_labels):
            print(f"{word}: {label}")
            
        # Return raw predictions for further analysis if needed
        return predictions, logits

# Run prediction with different settings
print("\n Standard Prediction (No Adjustments)")
predict_ner(sanskrit_sentences, use_threshold=False)

print("\n Threshold-Adjusted Prediction")
predict_ner(sanskrit_sentences, use_threshold=True, threshold_value=4.5)