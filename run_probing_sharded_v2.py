import torch
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
# ... (all other imports are the same) ...
import webdataset
from collections import defaultdict
import io
import re

def pth_decoder(data):
    """
    Decoder to load the *grouped* .pth file from raw bytes.
    The file contains a dict: {'activations': {...}}
    """
    # We must load to CPU, as the main process might be on GPU
    data_dict = torch.load(io.BytesIO(data), map_location="cpu")
    # Return the dictionary of tensors
    return data_dict['activations']

def load_data_for_one_probe(tar_path: Path, speakers_list: list, sub_key_to_probe: str = '__output__'):
    """
    Loads all data from ONE tar file (one probe).
    It will extract the specific tensor defined by `sub_key_to_probe`.
    """
    speaker_map = {name: i for i, name in enumerate(speakers_list)}
    
    X_data = []
    y_data = []
    
    try:
        dataset = webdataset.WebDataset(str(tar_path)).decode(pth_decoder).to_tuple("pth", "json")
        
        # 'activations' is now a dict like {'attn.to_q': tensor, ...}
        for activations, metadata in dataset:
            speaker_id = metadata['speaker_id']
            if speaker_id not in speaker_map:
                continue
                
            speaker_label = speaker_map[speaker_id]
            
            if sub_key_to_probe not in activations:
                # This can happen if a submodule wasn't run, etc.
                continue
                
            tensor = activations[sub_key_to_probe]
            
            # tensor is (B, N, D) -> [1, seq_len, dim]
            sampled_activations = tensor[0, ::5, :].float() # (num_samples, dim)
            
            if sampled_activations.shape[0] == 0:
                continue
                
            num_samples = sampled_activations.shape[0]
            labels = torch.full((num_samples,), fill_value=speaker_label, dtype=torch.long)
            
            X_data.append(sampled_activations)
            y_data.append(labels)
            
    except Exception as e:
        print(f"  Warning: Error loading {tar_path}: {e}")
        return None, None

    if not X_data:
        return None, None

    return torch.cat(X_data, dim=0).numpy(), torch.cat(y_data, dim=0).numpy()

def main():
    BASE_DIR = Path("./extracted_features_sharded")
    TRAIN_TARS_DIR = BASE_DIR / "train_tars"
    TEST_TARS_DIR = BASE_DIR / "test_tars"
    # We need the speaker list from *somewhere*.
    # Using the LibriTTS path is the most reliable way.
    LIBRITTS_ROOT = Path("/home/ewong/links/scratch/LibriTTS")
    SUBSET_TO_PROCESS = "test-clean"
    
    # --- Get Speaker List ---
    subset_dir = LIBRITTS_ROOT / SUBSET_TO_PROCESS
    speakers = sorted([p.name for p in subset_dir.iterdir() if p.is_dir()])
    print(f"Found {len(speakers)} speakers: {speakers}")
    
    results = defaultdict(dict)
    
    train_files = sorted(list(TRAIN_TARS_DIR.glob("*.tar")))
    if not train_files:
        print(f"Error: No .tar files found in {TRAIN_TARS_DIR}")
        return
        
    print(f"Found {len(train_files)} probe groups to test...")

    for train_tar_path in train_files:
        probe_filename = train_tar_path.name
        test_tar_path = TEST_TARS_DIR / probe_filename
        
        if not test_tar_path.exists():
            continue
            
        print(f"\n--- Probing Group: {probe_filename} ---")
        
        # Parse the filename to store results
        # "step_0005_group_transformer_blocks_21.tar"
        match = re.match(r"step_(\d+)_group_(.+)", train_tar_path.stem)
        if not match:
            print(f"  Skipping, weird filename: {probe_filename}")
            continue
            
        step = int(match.group(1))
        group_name = match.group(2).replace('_', '.') # 'transformer.blocks.21'

        # --- 1. Load Data for this probe ---
        # We'll probe the main output ('__output__') of each group
        # You could change this to 'attn.to_q' etc. if you wanted
        X_train, y_train = load_data_for_one_probe(train_tar_path, speakers, sub_key_to_probe='__output__')
        X_test, y_test = load_data_for_one_probe(test_tar_path, speakers, sub_key_to_probe='__output__')
        
        if X_train is None or X_test is None:
            print(f"  Skipping probe {group_name} @ {step}: No data loaded (was '__output__' captured?).")
            continue
            
        print(f"  Probing {group_name} ('__output__') @ step {step}")
        print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # --- 2. Train Probe ---
        probe = make_pipeline(
            StandardScaler(), 
            LogisticRegression(max_iter=100, solver='liblinear')
        )
        
        try:
            probe.fit(X_train, y_train)
        except ValueError as e:
            print(f"    ERROR training probe: {e}. Skipping.")
            continue

        # --- 3. Evaluate and Store ---
        accuracy = probe.score(X_test, y_test)
        print(f"  > Accuracy: {accuracy:.4f}")
        results[group_name][step] = accuracy

    # --- 4. Save and Plot Results ---
    print("\nProbing complete. Saving results...")
    
    results_df = pd.DataFrame(results).T.fillna(0)
    results_df.sort_index(axis=1, inplace=True) # Sort columns by step
    results_df.to_csv(BASE_DIR / "probe_accuracy_results_sharded.csv")
    
    plt.figure(figsize=(20, 15)) # Made taller for more layers
    sns.heatmap(
        results_df, 
        annot=True, 
        fmt=".2f", 
        cmap="viridis", 
        cbar_kws={'label': 'Test Accuracy'}
    )
    plt.title("Speaker ID Linear Probing Accuracy (Sharded & Grouped)")
    plt.xlabel("ODE Timestep Index")
    plt.ylabel("Model Layer Group")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "probe_accuracy_heatmap_sharded.png")
    
    print(f"\nDone! Results saved to {BASE_DIR}")

if __name__ == "__main__":
    main()