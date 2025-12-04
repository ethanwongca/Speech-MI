"""
Step 3: Build the Probing Metadata Table (Pandas Version)
"""
import json
import pandas as pd
from pathlib import Path

# Config
EXTRACTED_DIR = Path("./extracted_features_fixed_text")
TRAIN_INDEX = EXTRACTED_DIR / "train_index.json"
TEST_INDEX = EXTRACTED_DIR / "test_index.json"
OUTPUT_PATH = EXTRACTED_DIR / "probing_metadata.parquet"

def main():
    print(f"Loading indexes from {EXTRACTED_DIR}...")
    if not TRAIN_INDEX.exists() or not TEST_INDEX.exists():
        print("Error: Indexes not found. Run extraction first.")
        return

    with open(TRAIN_INDEX, 'r') as f: train_data = json.load(f)
    with open(TEST_INDEX, 'r') as f: test_data = json.load(f)

    print(f"Train samples: {len(train_data)}")
    print(f"Test samples:  {len(test_data)}")

    df_train = pd.DataFrame(train_data)
    df_train["split_mode"] = "train"
    df_test = pd.DataFrame(test_data)
    df_test["split_mode"] = "test"

    df_full = pd.concat([df_train, df_test], ignore_index=True)
    
    # Memory Optimization
    df_full["timestep"] = df_full["timestep"].astype("int32")
    df_full["speaker_id"] = df_full["speaker_id"].astype("category")
    df_full["block_name"] = df_full["block_name"].astype("category")
    df_full["split_mode"] = df_full["split_mode"].astype("category")

    print(f"Combined Metadata: {len(df_full)} rows.")
    df_full.to_parquet(OUTPUT_PATH)
    print(f"Saved metadata to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()