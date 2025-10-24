"""
Activation Collapse Script (v2 - with auto-cleanup)

Reads a 'step_XXXX' directory full of individual .pt files
and collapses them into logical "group" files (e.g., 'transformer_blocks.21.pt')
in a new 'collapsed/' subdirectory.

**Now with automatic cleanup:** Deletes the original 'step_XXXX'
directory after it is successfully collapsed.
"""

import torch
from pathlib import Path
import sys
import shutil  
from tqdm import tqdm

# --- Configuration ---
CAPTURE_DIR = Path("./capture_results/capture_EVERYTHING_all_steps")
# --- End Configuration ---

def get_group_and_key(hook_name: str) -> (str, str):
    """
    This is the SMART grouping logic.
    e.g., 'transformer_blocks.21.attn.to_q'
    returns -> ('transformer_blocks.21', 'attn.to_q')
    """
    parts = hook_name.split('.')

    if parts[0] == 'transformer_blocks' and len(parts) > 1 and parts[1].isdigit():
        # Group: 'transformer_blocks.21', Key: 'attn.to_q'
        group_name = f'transformer_blocks.{parts[1]}'
        sub_key = '.'.join(parts[2:]) if len(parts) > 2 else '__output__'
    
    elif '.' in hook_name:
        # Group: 'text_embed', Key: 'text_blocks.0'
        group_name = parts[0]
        sub_key = '.'.join(parts[1:])
    
    else:
        # Group: 'proj_out', Key: '__output__'
        group_name = hook_name
        sub_key = '__output__'
        
    return group_name, sub_key


# --- [!! NEW main() FUNCTION !!] ---
def main():
    if not CAPTURE_DIR.exists():
        print(f"Error: Capture directory not found at {CAPTURE_DIR}")
        sys.exit(1)

    print(f"Starting collapse for: {CAPTURE_DIR}")

    # Find all timestep directories (e.g., 'step_0000', 'step_0001', ...)
    timestep_dirs = sorted([d for d in CAPTURE_DIR.iterdir() if d.is_dir() and d.name.startswith('step_')])

    if not timestep_dirs:
        print("Error: No 'step_XXXX' directories found.")
        sys.exit(1)

    print(f"Found {len(timestep_dirs)} timesteps to process...")

    # Process each timestep directory
    for step_dir in (pbar := tqdm(timestep_dirs, desc="Timesteps")):
        
        # Wrap the entire process in a try/except block for safety.
        # We only delete if ALL steps for this timestep succeed.
        try:
            pbar.set_description(f"Processing {step_dir.name}")
            
            # Create the new output directory
            output_dir = CAPTURE_DIR / "collapsed" / step_dir.name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            grouped_tensors = {}
            
            all_files = list(step_dir.glob("*.pt"))
            if not all_files:
                print(f"Warning: No .pt files found in {step_dir.name}. Skipping.")
                continue

            # Loop through all 400+ files in this step directory
            for f_path in all_files:
                data = torch.load(f_path, map_location='cpu')
                hook_name = data['hook_name']
                tensor = data['tensor']
                
                group_name, sub_key = get_group_and_key(hook_name)
                
                if group_name not in grouped_tensors:
                    grouped_tensors[group_name] = {}
                
                grouped_tensors[group_name][sub_key] = tensor
            
            # Now save the new grouped files
            for group_name, tensor_dict in grouped_tensors.items():
                safe_filename = f"{group_name.replace('/', '_')}.pt"
                out_path = output_dir / safe_filename
                
                torch.save({
                    'group_name': group_name,
                    'activations': tensor_dict,
                }, out_path)

            # If we reached here, the collapse was successful.
            # Now, delete the original directory.
            pbar.set_description(f"Cleaning up {step_dir.name}")
            shutil.rmtree(step_dir)
            
        except Exception as e:
            print(f"\n--- [!! ERROR !!] ---")
            print(f"Failed to process or clean up {step_dir.name}: {e}")
            print(f"The original directory {step_dir.name} will NOT be deleted.")
            print(f"--- [!! END ERROR !!] ---\n")
            
    print("\n" + "="*70)
    print("COLLAPSE AND CLEANUP COMPLETE!")
    print(f"New grouped files are in: {CAPTURE_DIR / 'collapsed'}")
    print("Original 'step_XXXX' directories have been removed.")

if __name__ == "__main__":
    main()