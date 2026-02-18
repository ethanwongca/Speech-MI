"""
Step 1: Generate the reference dataset (Randomized & Two samples per speaker)
"""
import glob
from datasets import load_dataset, Audio, Dataset, Features, Value

# Config
DATA_PATH = "../libritts/data/**/*.parquet"
OUTPUT_DIR = "./libritts_reference_speakers"
SAMPLING_RATE = 24000
SAMPLES_PER_SPEAKER = 1  # Sample 0 = Train/Test, Sample 1 = Eval

def create_reference_dataset():
    files = glob.glob(DATA_PATH, recursive=True)
    if not files:
        print(f"Error: No files found at {DATA_PATH}")
        return

    print(f"Found {len(files)} parquet files. Starting Generator...")

    def speaker_generator():
        # Load parquet in streaming mode
        ds_stream = load_dataset("parquet", data_files=files, split="train", streaming=True)
        
        # random shuffling to randomly pick speaker vals
        # buffer_size ensures we aren't just taking the first speakers alphabetically/sequentially
        ds_stream = ds_stream.shuffle(seed=42, buffer_size=1000)
        
        ds_stream = ds_stream.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

        # Track count of samples collected per speaker
        speaker_counts = {}
        
        for sample in ds_stream:
            spk_id = str(sample["speaker_id"])
            
            # Skip if we already have enough samples for this speaker
            if speaker_counts.get(spk_id, 0) >= SAMPLES_PER_SPEAKER:
                continue

            try:
                audio_array = sample["audio"]["array"]
                
                # Filter: Ensure reference is long enough (>3s)
                if len(audio_array) > 72000:
                    current_count = speaker_counts.get(spk_id, 0)
                    
                    yield {
                        "speaker_id": spk_id,
                        "text_normalized": sample["text_normalized"],
                        "audio": {
                            "array": audio_array, 
                            "sampling_rate": SAMPLING_RATE
                        },
                        "sample_idx": current_count  # 0 or 1
                    }
                    
                    speaker_counts[spk_id] = current_count + 1
                    print(f"Speakers collected: {len(speaker_counts)} | Fully completed: {sum(1 for c in speaker_counts.values() if c >= SAMPLES_PER_SPEAKER)}", end="\r")

            except Exception:
                continue

    features = Features({
        "speaker_id": Value("string"),
        "text_normalized": Value("string"),
        "audio": Audio(sampling_rate=SAMPLING_RATE),
        "sample_idx": Value("int32")
    })

    print("\nGenerating dataset...")
    final_ds = Dataset.from_generator(speaker_generator, features=features)
    
    print(f"\nSaving to {OUTPUT_DIR}...")
    final_ds.save_to_disk(OUTPUT_DIR)
    print("Success.")

if __name__ == "__main__":
    create_reference_dataset()