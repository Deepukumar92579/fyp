class DCS:
    def __init__(self):
        self.sent_id = ""
        self.sentence = ""
        self.dcs_chunks = []
        self.lemmas = []
        self.cng = []


import pickle
import json
import glob
import os


def process_pickle_files():
    combined_data = {}
    pickle_files = glob.glob("*.p")

    for pickle_file in pickle_files:
        try:
            with open(pickle_file, "rb") as file:
                dcs_obj = pickle.load(file, encoding="utf-8")

            combined_data[dcs_obj.sent_id] = {
                "sentence": dcs_obj.sentence,
                "dcs_chunks": dcs_obj.dcs_chunks,
                "lemmas": dcs_obj.lemmas,
            }
            print(f"Processed: {pickle_file}")

        except Exception as e:
            print(f"Error processing {pickle_file}: {str(e)}")

    output_file = "combined_dcs_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

    print(f"\nProcessing complete. Data saved to {output_file}")
    print(f"Total files processed: {len(combined_data)}")


if __name__ == "__main__":
    process_pickle_files()
