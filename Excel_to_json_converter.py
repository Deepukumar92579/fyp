import pandas as pd
import json

# Step 1: Read the Excel file
df = pd.read_excel(r"/content/sanskrit_ner_results.xlsx", sheet_name="Sheet1")


# Step 2: Define a function to map tags to BIO format
def map_to_bio_tag(tag):
    if tag == "0":
        return "O"
    elif tag == "PERSON":
        return "B-PER"
    elif tag == "LOC":
        return "B-LOC"
    elif tag == "WEAPON":
        return "B-WEAPON"
    # Add more mappings if there are other entity types (e.g., 'TEXT' -> 'B-TEXT')
    else:
        return "O"  # Default to 'O' for unknown tags


# Step 3: Process the dataset
bio_data = []

# Iterate over each row in the DataFrame (skip the header row)
for index, row in df.iloc[1:].iterrows():
    # Extract segmented words and tags
    tokens = []
    ner_tags = []

    for i in range(2, 19, 2):  # Step by 2 to get word columns (C, E, G, ...)
        word = row[i]  # Word column (e.g., WORD_1, WORD_2, ...)
        tag = row[i + 1]  # Tag column (e.g., TAG_1, TAG_2, ...)

        # Only add the word and tag if the word is not NaN or empty
        if pd.notna(word) and word.strip() != "":
            tokens.append(word.strip())
            ner_tags.append(map_to_bio_tag(tag))

    # Create a dictionary for the current sentence
    sentence_data = {"tokens": tokens, "ner_tags": ner_tags}

    # Add to the list of sentences
    bio_data.append(sentence_data)

# Step 4: Save the BIO-formatted data to a JSON file
with open("sanskrit_ner_bio.json", "w", encoding="utf-8") as f:
    json.dump(bio_data, f, ensure_ascii=False, indent=4)

print("BIO-formatted dataset has been saved to 'sanskrit_ner_bio.json'.")
