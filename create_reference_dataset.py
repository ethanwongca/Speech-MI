"""
Step 1: Generate the reference dataset (Memory Safe & Feature Compliant)
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
        print(f"Error: No files found at {DATA_PATH}")
        return

    print(f"Found {len(files)} parquet files. Starting Generator...")

    # We use a generator to stream 1 file at a time (Low RAM usage)
    def speaker_generator():
        # Load parquet in streaming mode
        ds_stream = load_dataset("parquet", data_files=files, split="train", streaming=True)
        # Ensure we decode to get the array
        ds_stream = ds_stream.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

        seen_speakers = set()
        
        for sample in ds_stream:
            # Handle speaker_id as string (just in case it's int in parquet)
            spk_id = str(sample["speaker_id"])
            
            if spk_id in seen_speakers:
                continue

            try:
                audio_array = sample["audio"]["array"]
                
                # Filter: Ensure reference is long enough (>3s) for good cloning
                # 3s * 24000 Hz = 72,000 samples
                if len(audio_array) > 72000:
                    yield {
                        "speaker_id": spk_id,
                        "text_normalized": sample["text_normalized"],
                        # CRITICAL FIX: Wrap array in a dict for the Audio feature
                        "audio": {
                            "array": audio_array, 
                            "sampling_rate": SAMPLING_RATE
                        }
                    }
                    seen_speakers.add(spk_id)
                    print(f"Collected: {len(seen_speakers)} unique speakers", end="\r")

            except Exception:
                continue

    # Explicitly define features to match what we yield
    features = Features({
        "speaker_id": Value("string"),
        "text_normalized": Value("string"),
        "audio": Audio(sampling_rate=SAMPLING_RATE)
    })

    print("\nGenerating dataset...")
    final_ds = Dataset.from_generator(speaker_generator, features=features)
    
    print(f"\nSaving to {OUTPUT_DIR}...")
    final_ds.save_to_disk(OUTPUT_DIR)
    print("Success.")

if __name__ == "__main__":
    create_reference_dataset()