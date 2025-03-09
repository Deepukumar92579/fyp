from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Load a multilingual model fine-tuned for NER
model_name = "xlm-roberta-base"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# List of Sanskrit sentences for testing
sentences = [
    "माघवः पाण्डवः च एव दिव्यो शङ्खौ प्रदध्मतुः",
]

# Loop through sentences and perform NER
for sentence in sentences:
    print(f"\n🔹 Sentence: {sentence}")
    ner_results = ner_pipeline(sentence)

    # Print results
    for entity in ner_results:
        print(f"Entity: {entity['word']}, Label: {entity['entity_group']}, Score: {entity['score']:.2f}")