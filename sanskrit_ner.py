import json
import re
import pandas as pd
import unicodedata
from tqdm import tqdm

class SanskritNERProcessor:
    def __init__(self):
        # Entity gazetteers - expanded for Devanagari Sanskrit
        self.entity_dict = {
            "DEITY": [
                "शिव", "विष्णु", "ब्रह्मा", "इन्द्र", "अग्नि", "वायु", "वरुण", "कृष्ण", 
                "राम", "लक्ष्मी", "सरस्वती", "दुर्गा", "पार्वती", "गणेश", "हनुमान्",
                "सूर्य", "चन्द्र", "यम", "कुबेर", "महात्मन्", "आत्म", "हृषीकेश", "अच्युत",
                "गोविन्द", "मधुसूदन", "केशव"
            ],
            "PERSON": [
                "पाणिनि", "पतञ्जलि", "व्यास", "वाल्मीकि", "कालिदास", "भर्तृहरि", 
                "भास", "शङ्कर", "रामानुज", "मध्व", "चैतन्य", "शङ्कराचार्य", "योगिन्", "युक्त", "अयुक्त", "देही",
                "पाण्डव", "भीम", "अर्जुन", "युधिष्ठिर", "कुन्तीपुत्र", "नकुल", "सहदेव", "कौन्तेय", "पार्थ",
                "दुर्योधन", "भीष्म", "द्रोण", "कर्ण", "अश्वत्थामा", "धृतराष्ट्र", "द्रुपद", "धृष्टकेतु",
                "युयुधान", "विराट", "शिखण्डी", "सात्यकि", "कृप", "श्वशुर", "पितामह"
            ],
            "LOC": [
                "भारत", "गङ्गा", "हिमालय", "अयोध्या", "मथुरा", "द्वारका", 
                "काशी", "उज्जयिनी", "हस्तिनापुर", "इन्द्रप्रस्थ", "लङ्का", "कुरुक्षेत्र", "पुर", "लोक", "द्यावापृथिवी",
                "कुरु", "पृथिवी", "नभ"
            ],
            "TEXT": [
                "गीता", "वेद", "उपनिषद्", "पुराण", "महाभारत", "रामायण", "भगवद्गीता", "संधिविग्रह", "अन्वय"
            ],
            "CONCEPT": [
                "योग", "कर्म", "शान्ति", "ध्यान", "संन्यास", "फल", "संधिविग्रह", "अन्वय", "आत्मन्", "परमात्मा",
                "ब्रह्मचारि", "तप", "युक्त", "योगारूढ", "समबुद्धि", "श्रद्धा"
            ],
            "WEAPON": [
                "गाण्डीव", "शङ्ख", "चक्र", "गदा", "धनुष", "बाण", "शक्ति", "खड्ग", "पाञ्चजन्य", "देवदत्त"
            ]
        }
        
        # Common Sanskrit words that should not be tagged as entities
        self.common_words = [
            "च", "वा", "हि", "तु", "एव", "अपि", "इति", "तथा", "यथा", "सः", "साः", "तत्", 
            "अहम्", "त्वम्", "सर्व", "न", "इदम्", "तव", "मम", "अस्ति", "भवति", "न", "यदि", 
            "कुर्वन्ति", "आप्नोति", "कामकारेण", "मनसा", "सुखम्", "विग्रह", "अद्भुतम्", "उग्रम्",
            "अथ", "दृष्ट्वा", "तत्र", "तदा", "इह", "अत्र", "तद्", "येन", "तस्य", "अस्मिन्", "सह", "वै", "यत्", "ये"
        ]
    
    def segment_sanskrit_text(self, text):
        """Simple word segmentation for Sanskrit text"""
        # Basic segmentation by splitting on spaces and punctuation
        words = re.findall(r'\S+', text)
        return words
    
    def clean_text(self, text):
        """Remove special characters and normalize Unicode"""
        # Normalize Unicode to composed form (NFC)
        text = unicodedata.normalize('NFC', text)
        # Remove non-Devanagari characters except spaces
        text = re.sub(r'[^\u0900-\u097F\s]', '', text)
        return text
    
    def process_sentences(self, sentences):
        """Process a list of Sanskrit sentences and return NER annotations"""
        results = []
        
        for sentence in tqdm(sentences, desc="Processing sentences"):
            # Clean and prepare text
            cleaned_sentence = self.clean_text(sentence)
            # Segment words
            words = self.segment_sanskrit_text(cleaned_sentence)
            # Skip empty sentences
            if not words:
                continue
                
            # Tag entities
            tags = self.tag_entities(words)
            
            # Store result
            result = {
                "sentence": sentence,
                "words": words,
                "ner_tags": tags
            }
            results.append(result)
            
        return results
    
    def tag_entities(self, words):
        """Tag entities in a list of Sanskrit words"""
        tags = []
        
        for word in words:
            # Default tag is 'O' (Outside any entity)
            tag = "O"
            
            # Skip common words
            if word in self.common_words:
                tags.append(tag)
                continue
                
            # Check against entity dictionaries
            for entity_type, entity_list in self.entity_dict.items():
                if word in entity_list:
                    tag = entity_type
                    break
                    
                # Also check for partial matches (for compound words)
                for entity in entity_list:
                    if entity in word and len(entity) > 3:  # Minimum length to avoid false positives
                        tag = entity_type
                        break
                        
            tags.append(tag)
            
        return tags
    
    def save_to_json(self, results, output_file):
        """Save results to a JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(results)} annotated sentences to {output_file}")
        
    def export_to_conll(self, results, output_file):
        """Export results in CoNLL format for NER training"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                for word, tag in zip(result['words'], result['ner_tags']):
                    f.write(f"{word}\t{tag}\n")
                f.write("\n")  # Empty line between sentences
        print(f"Exported to CoNLL format: {output_file}")

# Enhanced processor for better Sanskrit NER
class EnhancedSanskritNER:
    def __init__(self):
        # Initialize base processor
        self.base_processor = SanskritNERProcessor()
        
        # Additional context-aware rules
        self.context_rules = {
            "योगिन": "PERSON",
            "युक्त": "PERSON",
            "आत्म": "CONCEPT",
            "कर्म": "CONCEPT",
            "फल": "CONCEPT",
            "गीता": "TEXT",
            "महात्मन्": "PERSON",
            "देही": "PERSON",
            "लोक": "LOC",
            "द्यावापृथिवी": "LOC",
            "पुर": "LOC",
            "परमात्मा": "CONCEPT",
            "ब्रह्मचारि": "CONCEPT",
            "योगी": "PERSON",
            "शङ्ख": "WEAPON",
            "गाण्डीव": "WEAPON",
            "पाञ्चजन्य": "WEAPON",
            "देवदत्त": "WEAPON"
        }
        
        # Special handling for Bhagavad Gita characters
        self.gita_characters = {
            "अर्जुन": "PERSON",
            "कृष्ण": "DEITY",
            "युधिष्ठिर": "PERSON",
            "भीम": "PERSON",
            "दुर्योधन": "PERSON",
            "धृतराष्ट्र": "PERSON",
            "पाण्डव": "PERSON",
            "कौन्तेय": "PERSON",
            "पार्थ": "PERSON",
            "गुडाकेश": "PERSON",
            "हृषीकेश": "DEITY",
            "अच्युत": "DEITY",
            "गोविन्द": "DEITY",
            "मधुसूदन": "DEITY",
            "केशव": "DEITY"
        }
        
    def process_text(self, text):
        """Process Sanskrit text with enhanced contextual rules"""
        # Split text into sentences by '।' and '.'
        sentences = re.split(r'[।.]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        all_results = []
        
        for sentence in sentences:
            words = self.base_processor.segment_sanskrit_text(sentence)
            tags = ["O"] * len(words)
            
            # Apply base tagging
            for i, word in enumerate(words):
                # Check against entity dictionaries
                for entity_type, entity_list in self.base_processor.entity_dict.items():
                    if word in entity_list:
                        tags[i] = entity_type
                        break
            
            # Apply context-specific rules
            for i, word in enumerate(words):
                # Apply Gita characters first
                for char_name, char_type in self.gita_characters.items():
                    if char_name in word:
                        tags[i] = char_type
                        break
                
                # Apply context rules if available
                for context_term, context_type in self.context_rules.items():
                    if context_term in word:
                        tags[i] = context_type
                        break
                
                # Special case for 'कर्म' followed by 'फल'
                if i < len(words) - 1 and "कर्म" in words[i] and "फल" in words[i+1]:
                    tags[i] = "CONCEPT"
                    tags[i+1] = "CONCEPT"
                
                # Skip common words
                if word in self.base_processor.common_words:
                    tags[i] = "O"
            
            result = {
                "sentence": sentence,
                "words": words,
                "ner_tags": tags
            }
            all_results.append(result)
        
        return all_results

    def process_excel_data(self, excel_file, unseg_col="Unseg", seg_col="Seg"):
        """Process Sanskrit text data from Excel with Unsegmented and Segmented columns"""
        try:
            df = pd.read_excel(excel_file)
        except:
            # Try CSV if Excel fails
            df = pd.read_csv(excel_file)
        
        print(f"Loaded data with {len(df)} rows")
        
        all_results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            unseg_text = str(row[unseg_col])
            seg_text = str(row[seg_col])
            
            # Clean and prepare text
            cleaned_unseg = self.base_processor.clean_text(unseg_text)
            cleaned_seg = self.base_processor.clean_text(seg_text)
            
            # Use segmented text for word boundaries
            words = self.base_processor.segment_sanskrit_text(cleaned_seg)
            
            # Skip empty entries
            if not words:
                continue
            
            # Tag entities
            tags = ["O"] * len(words)
            
            # Apply enhanced tagging
            for i, word in enumerate(words):
                # Skip common words
                if word in self.base_processor.common_words:
                    continue
                
                # Check Gita characters first
                for char_name, char_type in self.gita_characters.items():
                    if char_name in word:
                        tags[i] = char_type
                        break
                
                # Check against entity dictionaries
                if tags[i] == "O":  # If not already tagged
                    for entity_type, entity_list in self.base_processor.entity_dict.items():
                        if word in entity_list:
                            tags[i] = entity_type
                            break
                
                # Apply context rules if still not tagged
                if tags[i] == "O":
                    for context_term, context_type in self.context_rules.items():
                        if context_term in word and len(context_term) > 2:
                            tags[i] = context_type
                            break
            
            # Special case processing for common pairs
            for i in range(len(words) - 1):
                if "कर्म" in words[i] and "फल" in words[i+1]:
                    tags[i] = "CONCEPT"
                    tags[i+1] = "CONCEPT"
            
            result = {
                "unsegmented": unseg_text,
                "segmented": seg_text,
                "words": words,
                "ner_tags": tags
            }
            all_results.append(result)
        
        return all_results
    
    def save_to_excel(self, results, output_file):
        """Save results to an Excel file with entity highlighting"""
        rows = []
        for result in results:
            # Create a row dictionary
            row_dict = {
                "Unsegmented": result["unsegmented"],
                "Segmented": result["segmented"],
            }
            
            # Add each word and its tag
            for i, (word, tag) in enumerate(zip(result["words"], result["ner_tags"])):
                row_dict[f"Word_{i+1}"] = word
                row_dict[f"Tag_{i+1}"] = tag
            
            # Add entity summary
            entities = {}
            for word, tag in zip(result["words"], result["ner_tags"]):
                if tag != "O":
                    if tag not in entities:
                        entities[tag] = []
                    entities[tag].append(word)
            
            entity_summary = []
            for tag, words in entities.items():
                entity_summary.append(f"{tag}: {', '.join(words)}")
            
            row_dict["Entities"] = "; ".join(entity_summary)
            
            rows.append(row_dict)
        
        # Create DataFrame and save to Excel
        df = pd.DataFrame(rows)
        df.to_excel(output_file, index=False)
        print(f"Saved results to {output_file}")
        
        return df

# Function to process Excel file with Sanskrit text
def process_sanskrit_excel(excel_file, output_excel="sanskrit_ner_results.xlsx", 
                          output_json="sanskrit_ner_results.json"):
    """Process Sanskrit text data from Excel file with Unsegmented and Segmented columns"""
    processor = EnhancedSanskritNER()
    
    print(f"Processing file: {excel_file}")
    results = processor.process_excel_data(excel_file)
    
    # Save results
    if output_excel:
        df = processor.save_to_excel(results, output_excel)
    
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved results to {output_json}")
    
    # Print summary
    entity_counts = {}
    for result in results:
        for tag in result["ner_tags"]:
            if tag != "O":
                entity_counts[tag] = entity_counts.get(tag, 0) + 1
    
    print("\nEntity Statistics:")
    for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{entity_type}: {count}")
    
    return results

# Process example data from the provided text
def process_sample_data():
    """Process sample data from the provided examples"""
    # Create a temporary CSV file from the sample data
    sample_data = []
    
    # Data from paste-2.txt
    with open("paste-2.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for i in range(0, len(lines), 2):
        if i+1 < len(lines):
            unseg = lines[i].strip()
            seg = lines[i+1].strip()
            if unseg.startswith("Unseg"):  # Skip header
                continue
            sample_data.append({"Unseg": unseg, "Seg": seg})
    
    # Save to temporary CSV
    temp_df = pd.DataFrame(sample_data)
    temp_csv = "Cleaned_Gita.xlsx"
    temp_df.to_csv(temp_csv, index=False)
    
    # Process the CSV file
    return process_sanskrit_excel(temp_csv)

# Main entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sanskrit Named Entity Recognition")
    parser.add_argument("--excel", type=str, help="Path to Excel file with Unseg and Seg columns")
    parser.add_argument("--sample", action="store_true", help="Process sample data from the provided examples")
    
    args = parser.parse_args()
    
    if args.excel:
        results = process_sanskrit_excel(args.excel)
    elif args.sample:
        results = process_sample_data()
    else:
        print("No input specified. Using sample data.")
        results = process_sample_data()