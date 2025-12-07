"""
Step 3: Build the Probing Metadata Table (Train/Test/Eval)
"""
import json
import pandas as pd
from pathlib import Path

# Config
EXTRACTED_DIR = Path("./extracted_features_fixed_text")
TRAIN_INDEX = EXTRACTED_DIR / "train_index.json"
TEST_INDEX = EXTRACTED_DIR / "test_index.json"
EVAL_INDEX = EXTRACTED_DIR / "eval_index.json"
OUTPUT_PATH = EXTRACTED_DIR / "probing_metadata.parquet"

def main():
    print(f"Loading indexes from {EXTRACTED_DIR}...")
    
    frames = []
    
    # Load Train
    if TRAIN_INDEX.exists():
        with open(TRAIN_INDEX, 'r') as f: data = json.load(f)
        print(f"Train samples: {len(data)}")
        df = pd.DataFrame(data)
        df["split_mode"] = "train"
        frames.append(df)
        
    # Load Test
    if TEST_INDEX.exists():
        with open(TEST_INDEX, 'r') as f: data = json.load(f)
        print(f"Test samples:  {len(data)}")
        df = pd.DataFrame(data)
        df["split_mode"] = "test"
        frames.append(df)

    # Load Eval
    if EVAL_INDEX.exists():
        with open(EVAL_INDEX, 'r') as f: data = json.load(f)
        print(f"Eval samples:  {len(data)}")
        df = pd.DataFrame(data)
        df["split_mode"] = "eval"
        frames.append(df)

    if not frames:
        print("Error: No data found.")
        return

    df_full = pd.concat(frames, ignore_index=True)
    
    if "layers_included" in df_full.columns:
        print("Expanding condensed layers...")
        df_full = df_full.explode("layers_included")
        df_full = df_full.rename(columns={"layers_included": "block_name"})
    
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