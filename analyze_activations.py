#!/usr/bin/env python3
"""
F5-TTS Activation Analysis Script (v5)

Loads the "collapsed" grouped .pt files (e.g., 'transformer_blocks.21.pt')
for a single timestep. Each file contains a dictionary of tensors.

This script analyzes each tensor and saves a single, nested JSON
with the statistics for all groups.
"""

import torch
import json
from pathlib import Path
import sys
from tqdm import tqdm

# --- Configuration ---
# [!! MODIFIED !!] Point this to the NEW 'collapsed' directory
CAPTURE_DIR = Path("./capture_results/capture_EVERYTHING_all_steps/collapsed")

# Timestep to analyze.
TIMESTEP_TO_ANALYZE = "step_0000" 

# Model architecture details (must match your model)
NUM_HEADS = 16
HEAD_DIM = 64
HIDDEN_DIM = 1024 # Should be NUM_HEADS * HEAD_DIM
# --- End Configuration ---


# [!! RENAMED & MODIFIED !!]
def analyze_tensor_stats(tensor: torch.Tensor, num_heads: int, head_dim: int) -> dict:
    """
    Analyzes a single TENSOR (not a file path) and computes stats.
    """
    
    # Ensure tensor is on CPU and in float32 for stable stats
    tensor = tensor.cpu().to(torch.float32)

    # --- Basic stats (for all tensors) ---
    basic_stats = {
        "shape": list(tensor.shape),
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
    }

    # --- Head Analysis (only for 3D tensors of the right shape) ---
    if len(tensor.shape) != 3:
        return basic_stats # Not a candidate for head analysis
        
    B, S, H = tensor.shape
    if H != (num_heads * head_dim):
        return basic_stats # Not an attention tensor

    # --- This tensor IS a candidate, analyze heads ---
    try:
        # Reshape to expose the heads: [B, S, H] -> [B, S, NumHeads, HeadDim]
        tensor_reshaped = tensor.view(B, S, num_heads, head_dim)

        per_head_stats = {}
        for i in range(num_heads):
            head_tensor = tensor_reshaped[:, :, i, :]
            per_head_stats[f"head_{i}"] = {
                "mean": head_tensor.mean().item(),
                "std": head_tensor.std().item(),
                "min": head_tensor.min().item(),
                "max": head_tensor.max().item(),
                "norm": torch.linalg.norm(head_tensor).item()
            }
        
        return {
            **basic_stats,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "per_head_stats": per_head_stats
        }
    except Exception as e:
        print(f"    Warning: Head analysis failed for tensor with shape {tensor.shape}. Error: {e}")
        return basic_stats # Return basic stats on failure


# [!! HEAVILY MODIFIED !!]
def main():
    if not CAPTURE_DIR.exists():
        print(f"Error: 'collapsed' directory not found at {CAPTURE_DIR}")
        print("Please run the collapse_captures.py script first.")
        sys.exit(1)

    # Your new HookConfig saves into a sub-directory per timestep
    timestep_dir = CAPTURE_DIR / TIMESTEP_TO_ANALYZE
    if not timestep_dir.exists():
        print(f"Error: Timestep directory not found at {timestep_dir}")
        print(f"(Looked for {TIMESTEP_TO_ANALYZE} inside {CAPTURE_DIR})")
        sys.exit(1)

    print(f"Starting analysis of directory: {timestep_dir.name}")
    print("-" * 70)

    # Grab all 'group' .pt files from that timestep
    group_files = sorted(list(timestep_dir.glob("*.pt")))

    if not group_files:
        print(f"Error: No .pt group files found in {timestep_dir}")
        sys.exit(1)

    print(f"Found {len(group_files)} total group files to analyze...")
    
    # This will be our final NESTED results dictionary
    final_results = {
        "analysis_target": {
            "directory": str(timestep_dir),
            "timestep": TIMESTEP_TO_ANALYZE
        },
        "groups": {} # All group stats will go in here
    }

    # Iterate over each group file (e.g., 'transformer_blocks.0.pt')
    for f_path in tqdm(group_files, desc="Analyzing Groups"):
        try:
            # Load the group file
            data = torch.load(f_path, map_location='cpu')
            group_name = data['group_name']
            activations_dict = data['activations'] # This is the dict of tensors
            
            group_stats = {} # This will hold stats for this group
            
            # Iterate over tensors in this group (e.g., 'attn', 'attn.to_q')
            for sub_key, tensor in activations_dict.items():
                
                # Call the analysis function on the TENSOR itself
                stats = analyze_tensor_stats(tensor, NUM_HEADS, HEAD_DIM)
                group_stats[sub_key] = stats
            
            # Add this group's stats to the final result
            final_results["groups"][group_name] = group_stats

        except Exception as e:
            print(f"\nWarning: Failed to load or process {f_path.name}. Skipping. Error: {e}")
            continue

    # Save the final JSON
    # We save it in the PARENT of the 'collapsed' dir for easy access
    output_path = CAPTURE_DIR.parent / f"analysis_results_collapsed_{TIMESTEP_TO_ANALYZE}.json"
    
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print("-" * 70)
    print(f"\nAnalysis complete! NESTED results saved to:\n{output_path.absolute()}")

if __name__ == "__main__":
    main()