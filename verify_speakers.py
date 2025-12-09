"""
Verification Script: Checks exact speaker counts in the generated reference dataset
"""
from datasets import load_from_disk
from pathlib import Path

# Config matching your pipeline
INPUT_DATASET_PATH = "./libritts_reference_speakers"

def main():
    path = Path(INPUT_DATASET_PATH)
    if not path.exists():
        print(f"Error: Dataset not found at {path}. Run create_reference_dataset.py first.")
        return

    print(f"Loading dataset from {path}...")
    ds = load_from_disk(str(path))
    
    print(f"Total rows (audio files) in dataset: {len(ds)}")

    # Initialize counters
    train_speakers = set()  # sample_idx 0
    eval_speakers = set()   # sample_idx 1
    
    # Iterate and count
    for item in ds:
        spk_id = str(item['speaker_id'])
        idx = item['sample_idx']
        
        if idx == 0:
            train_speakers.add(spk_id)
        elif idx == 1:
            eval_speakers.add(spk_id)

    # --- THE VERIFICATION ---
    print("\n" + "="*40)
    print("VERIFICATION RESULTS")
    print("="*40)
    
    print(f"Unique Train Speakers (sample_idx=0): {len(train_speakers)}")
    print(f"Unique Eval Speakers  (sample_idx=1): {len(eval_speakers)}")
    
    # Check overlaps/missing
    missing_eval = len(train_speakers) - len(eval_speakers)
    
    if len(train_speakers) == 2429:
        print("\n Found exactly 2429 Train speakers.")
    else:
        print(f"\n Expected 2429, found {len(train_speakers)}.")

    if missing_eval == 34:
        print(f"Exactly 34 speakers are missing Eval samples (matches logs).")
    else:
        print(f"{missing_eval} speakers are missing Eval samples.")

if __name__ == "__main__":
    main()