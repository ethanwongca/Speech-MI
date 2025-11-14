#!/usr/bin/env python3
import torch
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tarfile
import io
from collections import defaultdict
import numpy as np

# ===================================================================
# HELPER FUNCTIONS FOR LOADING FROM INCREMENTAL TAR FORMAT
# ===================================================================

def load_tensor_from_tar_member(tar_file, member_name):
    """
    Extract and load a single .pth file from an open tar archive.
    
    Args:
        tar_file: Open tarfile.TarFile object
        member_name: Name of the member to extract (e.g., "1234/step_0005/transformer_blocks.0.pth")
    
    Returns:
        Tensor data or None if not found
    """
    try:
        member = tar_file.getmember(member_name)
        file_obj = tar_file.extractfile(member)
        if file_obj is None:
            return None
        
        # Load tensor (contains dict with 'activations' key from grouped saves)
        data_dict = torch.load(io.BytesIO(file_obj.read()), map_location="cpu")
        
        # Return the activations dict
        if 'activations' in data_dict:
            return data_dict['activations']
        else:
            return data_dict
            
    except KeyError:
        return None
    except Exception as e:
        print(f"    Warning: Failed to load {member_name}: {e}")
        return None


def get_tar_structure(tar_path: Path):
    """
    Scan a tar file to understand its structure.
    
    Returns:
        dict: {step_index: {layer_group: member_name}}
    """
    structure = defaultdict(lambda: defaultdict(str))
    
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.pth'):
                # Parse path: "speaker_id/step_0005/transformer_blocks.0.pth"
                parts = member.name.split('/')
                if len(parts) == 3:
                    speaker_id, step_folder, filename = parts
                    
                    # Extract step index
                    step_index = int(step_folder.replace('step_', ''))
                    
                    # Extract layer group name (remove .pth extension)
                    layer_group = filename.replace('.pth', '')
                    
                    structure[step_index][layer_group] = member.name
    
    return structure


def load_data_for_probe(
    tar_paths: list[Path],
    speakers_list: list[str],
    target_step: int,
    target_layer: str,
    sub_key: str = '__output__',
    sample_rate: int = 5
):
    """
    Load data from multiple tar files (one per speaker) for a specific layer and timestep.
    
    Args:
        tar_paths: List of tar file paths
        speakers_list: Ordered list of speaker IDs
        target_step: Which ODE timestep to probe
        target_layer: Which layer group to probe (e.g., 'transformer_blocks.0')
        sub_key: Which submodule output to use (e.g., '__output__', 'attn.to_q')
        sample_rate: Sample every N tokens from sequence
    
    Returns:
        X_data: numpy array of shape (total_samples, feature_dim)
        y_data: numpy array of speaker labels
    """
    speaker_map = {name: i for i, name in enumerate(speakers_list)}
    
    X_data = []
    y_data = []
    
    for tar_path in tar_paths:
        # Extract speaker ID from filename (e.g., "speaker_1234.tar")
        speaker_id = tar_path.stem.replace('speaker_', '')
        
        if speaker_id not in speaker_map:
            continue
        
        speaker_label = speaker_map[speaker_id]
        
        try:
            with tarfile.open(tar_path, 'r') as tar:
                # Construct expected path in tar
                member_name = f"{speaker_id}/step_{target_step:04d}/{target_layer}.pth"
                
                activations = load_tensor_from_tar_member(tar, member_name)
                
                if activations is None:
                    continue
                
                # activations is a dict like {'__output__': tensor, 'attn.to_q': tensor, ...}
                if sub_key not in activations:
                    continue
                
                tensor = activations[sub_key]
                
                # tensor shape: (batch, seq_len, feature_dim)
                # For probing, sample every `sample_rate` tokens
                if tensor.ndim == 3:
                    sampled = tensor[0, ::sample_rate, :].float()
                elif tensor.ndim == 2:
                    sampled = tensor[::sample_rate, :].float()
                else:
                    print(f"    Warning: Unexpected tensor shape {tensor.shape} for {speaker_id}")
                    continue
                
                if sampled.shape[0] == 0:
                    continue
                
                num_samples = sampled.shape[0]
                labels = torch.full((num_samples,), fill_value=speaker_label, dtype=torch.long)
                
                X_data.append(sampled)
                y_data.append(labels)
                
        except Exception as e:
            print(f"    Warning: Error loading {tar_path.name}: {e}")
            continue
    
    if not X_data:
        return None, None
    
    return torch.cat(X_data, dim=0).numpy(), torch.cat(y_data, dim=0).numpy()


# ===================================================================
# MAIN PROBING SCRIPT
# ===================================================================

def main():
    # --- Configuration ---
    BASE_DIR = Path("./extracted_features_sharded")
    TRAIN_TARS_DIR = BASE_DIR / "train_tars"
    TEST_TARS_DIR = BASE_DIR / "test_tars"
    
    LIBRITTS_ROOT = Path("/home/ewong/links/scratch/LibriTTS")
    SUBSET_TO_PROCESS = "test-clean"
    
    SAMPLE_RATE = 5  # Sample every 5 tokens
    TIMESTEPS_TO_PROBE = [0, 5, 10, 15, 20, 25, 31]  # Every ~5 timesteps
    
    # --- Get Speaker List ---
    subset_dir = LIBRITTS_ROOT / SUBSET_TO_PROCESS
    speakers = sorted([p.name for p in subset_dir.iterdir() if p.is_dir()])
    print(f"Found {len(speakers)} speakers")
    
    # --- Get Tar Files ---
    train_tars = sorted(list(TRAIN_TARS_DIR.glob("speaker_*.tar")))
    test_tars = sorted(list(TEST_TARS_DIR.glob("speaker_*.tar")))
    
    if not train_tars or not test_tars:
        print("Error: No tar files found!")
        return
    
    print(f"Found {len(train_tars)} train tars, {len(test_tars)} test tars")
    
    # --- Discover Available Layers ---
    print("\nScanning first tar to discover layer structure...")
    sample_structure = get_tar_structure(train_tars[0])
    
    if not sample_structure:
        print("Error: Could not parse tar structure!")
        return
    
    # Get all unique layer groups across all timesteps
    all_layers = set()
    for step_dict in sample_structure.values():
        all_layers.update(step_dict.keys())
    
    all_layers = sorted(list(all_layers))
    print(f"Found {len(all_layers)} unique layer groups")
    print(f"Available timesteps: {sorted(sample_structure.keys())}")
    
    # --- Probe Each Layer at Each Timestep ---
    results = defaultdict(dict)
    
    total_experiments = len(all_layers) * len(TIMESTEPS_TO_PROBE)
    experiment_count = 0
    
    for layer_group in all_layers:
        print(f"\n{'='*70}")
        print(f"Probing layer: {layer_group}")
        print(f"{'='*70}")
        
        for timestep in TIMESTEPS_TO_PROBE:
            experiment_count += 1
            print(f"\n[{experiment_count}/{total_experiments}] Layer: {layer_group}, Timestep: {timestep}")
            
            # Load training data
            print("  Loading train data...")
            X_train, y_train = load_data_for_probe(
                tar_paths=train_tars,
                speakers_list=speakers,
                target_step=timestep,
                target_layer=layer_group,
                sub_key='__output__',
                sample_rate=SAMPLE_RATE
            )
            
            # Load test data
            print("  Loading test data...")
            X_test, y_test = load_data_for_probe(
                tar_paths=test_tars,
                speakers_list=speakers,
                target_step=timestep,
                target_layer=layer_group,
                sub_key='__output__',
                sample_rate=SAMPLE_RATE
            )
            
            if X_train is None or X_test is None:
                print(f"  ⚠️  No data found - skipping")
                continue
            
            print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
            
            # Train linear probe
            probe = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=200, solver='liblinear', random_state=42)
            )
            
            try:
                probe.fit(X_train, y_train)
                accuracy = probe.score(X_test, y_test)
                
                print(f"  ✓ Accuracy: {accuracy:.4f}")
                results[layer_group][timestep] = accuracy
                
            except Exception as e:
                print(f"  ✗ Error training probe: {e}")
                continue
    
    # --- Save Results ---
    print("\n" + "="*70)
    print("PROBING COMPLETE - SAVING RESULTS")
    print("="*70)
    
    if not results:
        print("Error: No results to save!")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results).T.fillna(0)
    results_df = results_df.reindex(sorted(results_df.columns), axis=1)  # Sort timesteps
    results_df = results_df.sort_index()  # Sort layers
    
    # Save CSV
    csv_path = BASE_DIR / "probe_accuracy_results.csv"
    results_df.to_csv(csv_path)
    print(f"✓ Saved results to {csv_path}")
    
    # Create heatmap
    plt.figure(figsize=(14, max(10, len(all_layers) * 0.3)))
    sns.heatmap(
        results_df,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={'label': 'Test Accuracy'},
        vmin=0,
        vmax=1
    )
    plt.title(f"Speaker ID Linear Probing Accuracy\n({len(speakers)} speakers, sampling every {SAMPLE_RATE} tokens)")
    plt.xlabel("ODE Timestep")
    plt.ylabel("Layer Group")
    plt.tight_layout()
    
    heatmap_path = BASE_DIR / "probe_accuracy_heatmap.png"
    plt.savefig(heatmap_path, dpi=150)
    print(f"✓ Saved heatmap to {heatmap_path}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Best layer overall: {results_df.max(axis=1).idxmax()} (accuracy: {results_df.max(axis=1).max():.4f})")
    print(f"Best timestep overall: {results_df.max(axis=0).idxmax()} (accuracy: {results_df.max(axis=0).max():.4f})")
    print(f"\nMean accuracy per timestep:")
    print(results_df.mean(axis=0).to_string())
    print(f"\nMean accuracy per layer (top 10):")
    print(results_df.mean(axis=1).nlargest(10).to_string())
    
    print("\n✓ All done!")


if __name__ == "__main__":
    main()