import json
import pandas as pd
from tqdm import tqdm

# Sanskrit entity dictionaries
known_entities = {
    "DEITY": [
        "śiva", "viṣṇu", "brahmā", "indra", "agni", "vāyu", "varuṇa", "kṛṣṇa", 
        "rāma", "lakṣmī", "sarasvatī", "durgā", "pārvatī", "gaṇeśa", "hanumān",
        "sūrya", "candra", "yama", "kubera", "kārttikeya", "māhendra", "kuliśa"
    ],
    "PERSON": [
        "pāṇini", "patañjali", "vyāsa", "vālmīki", "kālidāsa", "bhartṛhari", 
        "bhāsa", "śaṅkara", "rāmānuja", "madhva", "caitanya", "śaṅkarācārya"
    ],
    "LOC": [
        "bhārata", "gaṅgā", "himālaya", "ayodhyā", "mathurā", "dvārakā", 
        "kāśī", "ujjayinī", "hastināpura", "indraprastha", "laṅkā", "kurukṣetra"
    ],
    "WORK": [
        "veda", "upaniṣad", "purāṇa", "mahābhārata", "rāmāyaṇa", "bhagavadgītā",
        "manusmṛti", "yogasūtra", "brahmasūtra", "pañcatantra", "hitopadeśa"
    ],
    "ELEMENT": [
        "agni", "jala", "vāyu", "pṛthvī", "ākāśa"
    ],
    "CLASS": [
        "brahman", "kṣatriya", "viś", "śūdra"
    ]
}

class SanskritNERAnnotator:
    def __init__(self, input_file, output_file, gazetteer=None):
        self.input_file = input_file
        self.output_file = output_file
        self.gazetteer = gazetteer or known_entities
        self.data = self.load_data()
        self.annotated_data = {}
        
    def load_data(self):
        with open(self.input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_data(self):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.annotated_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(self.annotated_data)} annotated sentences to {self.output_file}")
    
    def automatic_annotation(self):
        """Apply gazetteer-based automatic annotation"""
        for key, entry in tqdm(self.data.items(), desc="Annotating"):
            ner_tags = ["O"] * len(entry["dcs_chunks"])
            
            # Apply gazetteer matching
            for i, word in enumerate(entry["dcs_chunks"]):
                for entity_type, entities in self.gazetteer.items():
                    word_lower = word.lower()
                    if word_lower in entities:
                        ner_tags[i] = entity_type
                        break
                    
                    # Special case for kṛṣṇa - could be a color or a deity
                    if word_lower == "kṛṣṇa":
                        # Check surrounding context
                        if i > 0 and entry["dcs_chunks"][i-1].lower() in ["śrī", "bhagavān"]:
                            ner_tags[i] = "DEITY"
                        else:
                            # Default to deity unless we have evidence it's a color
                            if "color" in entry["sentence"].lower() or any(c in entry["sentence"].lower() for c in ["śveta", "rakta", "pīta"]):
                                ner_tags[i] = "COLOR"
                            else:
                                ner_tags[i] = "DEITY"
            
            # Keep the original entry structure and add ner_tags
            annotated_entry = entry.copy()
            annotated_entry["ner_tags"] = ner_tags
            self.annotated_data[key] = annotated_entry
    
    def interactive_annotation(self, sample_keys=None):
        """Interactive annotation for specified keys or all data"""
        keys_to_annotate = sample_keys if sample_keys else list(self.data.keys())
        
        for key in tqdm(keys_to_annotate, desc="Manual annotation"):
            entry = self.data[key]
            print(f"\nKey: {key}, Sentence: {entry['sentence']}")
            ner_tags = []
            
            for i, word in enumerate(entry["dcs_chunks"]):
                # First check in gazetteer for suggestion
                suggested_tag = "O"
                for entity_type, entities in self.gazetteer.items():
                    if word.lower() in entities:
                        suggested_tag = entity_type
                        break
                
                valid_input = False
                while not valid_input:
                    user_input = input(f"Word: {word} [Suggested: {suggested_tag}]\n"
                                      f"Enter tag (DEITY, PERSON, LOC, ORG, WORK, TIME, ELEMENT, CLASS, COLOR, O) or press Enter for suggestion: ")
                    
                    if user_input == "":
                        tag = suggested_tag
                        valid_input = True
                    elif user_input.upper() in ["DEITY", "PERSON", "LOC", "ORG", "WORK", "TIME", "ELEMENT", "CLASS", "COLOR", "O"]:
                        tag = user_input.upper()
                        valid_input = True
                    else:
                        print("Invalid tag. Please try again.")
                
                ner_tags.append(tag)
            
            # Preserve the original entry and add ner_tags
            annotated_entry = entry.copy()
            annotated_entry["ner_tags"] = ner_tags
            self.annotated_data[key] = annotated_entry
    
    def bulk_annotation(self):
        """Process the entire dataset and apply annotations based on rules and gazetteers"""
        for key, entry in tqdm(self.data.items(), desc="Processing"):
            chunks = entry["dcs_chunks"]
            ner_tags = ["O"] * len(chunks)
            
            # Apply entity recognition rules
            for i, word in enumerate(chunks):
                word_lower = word.lower()
                
                # Check against gazetteers
                entity_found = False
                for entity_type, entities in self.gazetteer.items():
                    if word_lower in entities:
                        ner_tags[i] = entity_type
                        entity_found = True
                        break
                
                # Apply contextual rules if no entity was found
                if not entity_found:
                    # Colors
                    if word_lower in ["śveta", "rakta", "pīta", "kṛṣṇa"] and any(c in chunks for c in ["śveta", "rakta", "pīta"]):
                        ner_tags[i] = "COLOR"
                    
                    # Natural elements
                    elif word_lower in ["agni", "sarpa"]:
                        # Check context - if it's about fears/dangers, it's not an element deity
                        if "bhaya" in chunks or "mṛtyu" in chunks:
                            ner_tags[i] = "ELEMENT"
                        else:
                            ner_tags[i] = "DEITY"
                            
                    # Māhendra - Indra's attribute or a gem
                    elif word_lower == "māhendra" and "maṇi" in chunks:
                        ner_tags[i] = "GEM"
                        
                    # Special context for social classes
                    elif word_lower in ["brahman", "kṣatriya", "viś", "śūdra"]:
                        if all(c in chunks for c in ["brahman", "kṣatriya", "viś", "śūdra"]):
                            ner_tags[i] = "CLASS"
            
            # Create annotated entry preserving original structure
            annotated_entry = entry.copy()
            annotated_entry["ner_tags"] = ner_tags
            self.annotated_data[key] = annotated_entry

# Process your specific examples
def process_examples(example_data):
    # Create a temporary dictionary to simulate loading from a file
    with open("temp_data.json", "w", encoding="utf-8") as f:
        json.dump(example_data, f, ensure_ascii=False, indent=2)
    
    annotator = SanskritNERAnnotator(
        input_file="temp_data.json",
        output_file="sanskrit_ner_annotated.json"
    )
    
    # Add additional entity types specific to these examples
    annotator.gazetteer["COLOR"] = ["śveta", "rakta", "pīta", "kṛṣṇa"]
    annotator.gazetteer["GEM"] = ["maṇi", "kuliśa"]
    
    # Run automatic annotation with specific rules
    annotator.bulk_annotation()
    
    # Return the annotated data
    return annotator.annotated_data

# Example usage
if __name__ == "__main__":
    # Replace with your actual data loading
    with open("sanskrit_segmented_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    annotated_data = process_examples(data)
    
    # Save the results
    with open("sanskrit_ner_annotated.json", "w", encoding="utf-8") as f:
        json.dump(annotated_data, f, ensure_ascii=False, indent=2)
    
    print("NER annotation completed successfully!")