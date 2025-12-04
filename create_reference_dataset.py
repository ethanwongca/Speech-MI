"""
Step 1: Generate the reference dataset
"""

import glob
from datasets import load_dataset, Audio, Dataset, Features, Value

# --- CONFIGURATION ---
DATA_PATH = "../libritts/data/**/*.parquet"
OUTPUT_DIR = "./libritts_reference_speakers"
SAMPLING_RATE = 24000

def create_reference_dataset():
    files = glob.glob(DATA_PATH, recursive=True)
    if not files:
        print("Error: No files found.")
        return

    print(f"Found {len(files)} parquet files. Starting Generator...")

    # This function yields one sample at a time, preventing RAM explosion.
    def speaker_generator():
        # Initialize streaming inside the generator
        ds_stream = load_dataset("parquet", data_files=files, split="train", streaming=True)
        ds_stream = ds_stream.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

        seen_speakers = set()
        
        for sample in ds_stream:
            spk_id = str(sample["speaker_id"])
            
            if spk_id in seen_speakers:
                continue

            try:
                audio_array = sample["audio"]["array"]
                # Ensure reference is long enough (>3s)
                if len(audio_array) > 72000:
                    yield {
                        "speaker_id": spk_id,
                        "text_normalized": sample["text_normalized"], 
                        "audio": audio_array
                    }
                    seen_speakers.add(spk_id)
                    print(f"Collected: {len(seen_speakers)} unique speakers", end="\r")

            except Exception:
                continue

    # Define features to ensure 'audio' is saved correctly as Audio type
    features = Features({
        "speaker_id": Value("string"),
        "text_normalized": Value("string"),
        "audio": Audio(sampling_rate=SAMPLING_RATE)
    })

    # Create dataset from generator (Writes to disk incrementally)
    print("Generating dataset (memory safe)...")
    final_ds = Dataset.from_generator(speaker_generator, features=features)
    
    print("\nSaving final dataset to disk...")
    final_ds.save_to_disk(OUTPUT_DIR)
    print("Success.")

if __name__ == "__main__":
    create_reference_dataset()