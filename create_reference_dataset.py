"""
Step 1: Create Reference Dataset
"""
import glob
from datasets import load_dataset, Audio, Dataset

# --- CONFIGURATION ---
DATA_PATH = "../libritts/data/**/*.parquet"
OUTPUT_DIR = "./libritts_reference_speakers"
SAMPLING_RATE = 24000

def create_reference_dataset():
    files = glob.glob(DATA_PATH, recursive=True)
    if not files:
        print("Error: No files found.")
        return

    print(f"Found {len(files)} parquet files. Starting Stream...")

    # Load in Streaming Mode
    ds_stream = load_dataset("parquet", data_files=files, split="train", streaming=True)
    ds_stream = ds_stream.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    valid_samples = []
    seen_speakers = set()
    
    print("Iterating to find 1 clean reference per speaker...")
    
    for i, sample in enumerate(ds_stream):
        spk_id = str(sample["speaker_id"])
        
        if spk_id in seen_speakers:
            continue

        try:
            audio_array = sample["audio"]["array"]
            # Ensure reference is long enough to clone well (>3s is best for F5)
            if len(audio_array) > 72000:
                
                valid_samples.append({
                    "speaker_id": spk_id,
                    "text_normalized": sample["text_normalized"], 
                    "audio": audio_array
                })
                
                seen_speakers.add(spk_id)
                print(f"Collected: {len(valid_samples)} unique speakers", end="\r")

        except Exception as e:
            continue

    print(f"\nScanning complete. Found {len(valid_samples)} total unique speakers.")

    print(f"Saving to {OUTPUT_DIR}...")
    final_ds = Dataset.from_list(valid_samples)
    final_ds = final_ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    final_ds.save_to_disk(OUTPUT_DIR)
    print("Success.")

if __name__ == "__main__":
    create_reference_dataset()