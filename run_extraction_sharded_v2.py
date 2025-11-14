#!/usr/bin/env python3
"""
FIXED EXTRACTION SCRIPT WITH PROPER MEMORY MANAGEMENT
=====================================================

KEY FIXES APPLIED:
1. Proper tensor cleanup after capture
2. Correct hook lifecycle management
3. Memory-safe packing with immediate tensor deletion
4. Proper garbage collection timing
5. Fixed metadata handling
6. CHUNKED CAPTURE to avoid OOM killer when hooking all layers.

"""

from collections import defaultdict
import torch
import json
from pathlib import Path
import shutil
import webdataset
import time
import io
import torchaudio
import sys
from omegaconf import OmegaConf
from hydra.utils import get_class
from importlib.resources import files
import gc

# --- Core Imports from F5-TTS ---
from f5_tts.infer.utils_infer import (
    load_model, load_vocoder, preprocess_ref_audio_text, infer_process
)
from f5_tts.model.cfm import HookConfig, HookManager, CFM 
from f5_tts.model.backbones.dit import DiT 

# --- ================================================ ---
# --- Helper Functions
# --- ================================================ ---

def load_f5_model_and_vocoder(device="cuda"):
    """Loads the F5-TTS model and vocoder using your paths."""
    print(f"Device: {device}")
    
    model_name = "F5TTS_v1_Base"
    
    LOCAL_MODEL_PATH = "/home/ewong/links/scratch/checkpoints/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    LOCAL_VOCODER_PATH = "/home/ewong/links/scratch/checkpoints/vocos-mel-24khz"
    
    if not Path(LOCAL_MODEL_PATH).exists():
        raise FileNotFoundError(f"Model checkpoint not found at {LOCAL_MODEL_PATH}")
    if not Path(LOCAL_VOCODER_PATH).exists():
        raise FileNotFoundError(f"Vocoder not found at {LOCAL_VOCODER_PATH}")

    print("Loading vocoder...")
    vocoder = load_vocoder(
        vocoder_name="vocos", is_local=True, local_path=LOCAL_VOCODER_PATH, device=device
    )
    print("Vocoder loaded\n")
    
    print("Loading F5-TTS model...")
    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model_name}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arch = model_cfg.model.arch
    
    model = load_model(
        model_cls, model_arch, LOCAL_MODEL_PATH,
        mel_spec_type="vocos", vocab_file="", device=device
    )
    print("Model loaded\n")
    
    if not hasattr(model, 'transformer'):
        raise AttributeError("Model has no 'transformer' attribute! Hooking will fail.")
        
    return model, vocoder


# --- ================================================ ---
# --- FIXED: Per-Timestep Shard Writing
# --- ================================================ ---

class PerTimestepShardWriter:
    """
    Writes WebDataset shards immediately after each timestep.
    Creates one writer per timestep, writes all groups, then immediately closes.
    
    FIX: Added proper error handling and memory cleanup
    """
    
    def __init__(self, output_dir: Path, speaker_id: str, split: str):
        """
        Args:
            output_dir: Base directory for shards (e.g., ./extracted_features_sharded/train_tars)
            speaker_id: Current speaker ID
            split: "train" or "test"
        """
        self.output_dir = output_dir
        self.speaker_id = speaker_id
        self.split = split
        self.samples_written = 0
        
    def write_timestep(self, step_index: int, step_dir: Path, group_names: list[str]):
        """
        Writes all groups for a single timestep, then immediately closes the writer.
        
        Args:
            step_index: Timestep index
            step_dir: Path to directory containing grouped tensor files
            group_names: List of layer group names to pack
        """
        # Create a UNIQUE shard pattern for this specific timestep
        # Format: speaker_12345_step_0001_*.tar
        shard_pattern = str(self.output_dir / f"speaker_{self.speaker_id}_step_{step_index:04d}_%s.tar")
        
        # Create a NEW writer for just this timestep
        writer = webdataset.ShardWriter(shard_pattern)
        
        groups_written = 0
        
        try:
            for group_name in group_names:
                safe_filename = f"{group_name.replace('/', '_').replace('.', '_')}.pt"
                tensor_path = step_dir / safe_filename
                
                if not tensor_path.exists():
                    print(f"    Warning: Missing grouped tensor {tensor_path}")
                    continue

                try:
                    # FIX: Read bytes directly without loading tensor into memory
                    tensor_data_bytes = tensor_path.read_bytes()
                except Exception as e:
                    print(f"    Warning: Failed to load {tensor_path}: {e}")
                    continue

                # Metadata for this sample
                sample_metadata = {
                    'speaker_id': self.speaker_id,
                    'step_index': step_index,
                    'layer_group': group_name,
                    'split': self.split,
                }
                sample_metadata_data = json.dumps(sample_metadata).encode('utf-8')
                
                # Create shard key (each group gets its own shard)
                shard_key = f"group_{safe_filename.split('.')[0]}"
                
                # Sample key includes speaker and step for uniqueness
                sample_key = f"{self.speaker_id}_step{step_index:04d}"
                
                sample = {
                    "__key__": sample_key,
                    "pth": tensor_data_bytes,
                    "json": sample_metadata_data
                }
                
                writer.write(sample, shard=shard_key)
                self.samples_written += 1
                groups_written += 1
                
                # FIX: Delete bytes immediately after writing
                del tensor_data_bytes
                del sample
                
        except Exception as e:
            print(f"    ERROR writing timestep {step_index}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # CRITICAL: Close writer immediately after this timestep
            try:
                writer.close()
            except Exception as e:
                print(f"    Warning: Error closing writer: {e}")
            
            # Force garbage collection
            del writer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"    ✓ Timestep {step_index} packed and flushed to disk ({groups_written}/{len(group_names)} groups)")


def pack_speaker_incrementally(
    temp_dir: Path,
    shard_writer: PerTimestepShardWriter,
    speaker_id: str
) -> int:
    """
    Reads metadata and packs timesteps ONE AT A TIME with immediate flushing.
    
    FIX: Added better error handling and progress tracking
    
    Returns:
        Total number of samples written
    """
    metadata_path = temp_dir / "capture_metadata.json"
    if not metadata_path.exists():
        print(f"  Warning: No metadata file found at {metadata_path}")
        return 0

    try:
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
    except Exception as e:
        print(f"  Error reading metadata: {e}")
        return 0

    timesteps = meta.get('timesteps', [])
    if not timesteps:
        print(f"  Warning: No timesteps found in metadata")
        return 0

    print(f"  Packing {len(timesteps)} timesteps incrementally...")
    
    packed_count = 0
    for idx, timestep_info in enumerate(timesteps):
        step_index = timestep_info.get('step_index')
        step_dir = Path(timestep_info.get('directory', ''))
        group_names = timestep_info.get('group_names', [])
        
        if not step_dir.exists():
            print(f"    Warning: Step directory {step_dir} not found, skipping")
            continue
        
        # Write this timestep and immediately flush to disk
        try:
            shard_writer.write_timestep(step_index, step_dir, group_names)
            packed_count += 1
        except Exception as e:
            print(f"    Error packing step {step_index}: {e}")
            continue
        
        # FIX: Delete the timestep directory immediately after packing
        try:
            shutil.rmtree(step_dir, ignore_errors=True)
        except Exception as e:
            print(f"    Warning: Could not delete {step_dir}: {e}")
        
        # FIX: Periodic garbage collection
        if (idx + 1) % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    print(f"  Successfully packed {packed_count}/{len(timesteps)} timesteps")
    return shard_writer.samples_written


# --- ================================================ ---
# --- REPLACED: PHASE 1 - CAPTURE (with Chunking)
# --- ================================================ ---

def run_speaker_capture(
    model, 
    vocoder,
    speaker_id: str,
    wav_path: Path,
    text_prompt: str,
    temp_dir_base: Path,
    inference_params: dict,
    device: str,
    run_type: str = "capture"
) -> bool:
    """
    Runs hook extraction by breaking layers into chunks to avoid OOM.
    Runs inference N times per audio, each time capturing one chunk of layers.
    """
    print(f"\n--- PHASE 1: {run_type.upper()} Capture for Speaker {speaker_id} ({wav_path.name}) ---")
    
    run_temp_dir = temp_dir_base / speaker_id
    run_temp_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Get ALL layers first ---
    all_module_paths = []
    if not hasattr(model, 'transformer'):
        raise AttributeError("Model has no 'transformer' attribute! Hooking will fail.")
        
    for module_path, module in model.transformer.named_modules():
        if not module_path: 
            continue
        all_module_paths.append(module_path)
        
    if not all_module_paths:
        print("  Error: No modules found in model.transformer. Skipping.")
        return False
        
    print(f"  Found {len(all_module_paths)} total layers to capture.")

    # --- 2. Define chunk size and create chunks ---
    # YOU CAN TUNE THIS. If 50 still OOMs, try 25.
    CHUNK_SIZE = 50 
    layer_chunks = [
        all_module_paths[i:i + CHUNK_SIZE] 
        for i in range(0, len(all_module_paths), CHUNK_SIZE)
    ]
    print(f"  Splitting into {len(layer_chunks)} chunks of ~{CHUNK_SIZE} layers each.")

    # --- 3. Preprocess audio and text ONCE ---
    try:
        ref_audio, ref_text = preprocess_ref_audio_text(str(wav_path), text_prompt)
    except Exception as e:
        print(f"  Skipping speaker {speaker_id}: error preprocessing audio/text: {e}")
        return False
    
    # --- 4. Loop over chunks and run inference for each ---
    all_chunks_success = True
    for i, chunk_paths in enumerate(layer_chunks):
        
        print(f"\n    --- Running Capture Chunk {i+1}/{len(layer_chunks)} ({len(chunk_paths)} layers) ---")
        
        # 1. Define layers for THIS chunk
        layers_to_capture_chunk = {path: path for path in chunk_paths}
        
        # 2. Create HookConfig and Manager for THIS chunk
        hook_config = HookConfig(
            capture_all_layers=False, 
            capture_layers=layers_to_capture_chunk,
            save_per_timestep=True,
            time_filter=lambda t, step: True,
            clear_activations_per_step=True,
            precision=torch.bfloat16,
            save_dir=str(run_temp_dir),
            device=device   
        )
        hook_manager = HookManager(config=hook_config)

        # 3. Run inference
        try:
            result = infer_process(
                ref_audio=ref_audio,
                ref_text=ref_text,
                gen_text=ref_text,        
                model_obj=model,
                vocoder=vocoder,
                hook_manager=hook_manager,  
                **inference_params
            )
            if result is None:
                print(f"    Warning: Capture chunk {i+1} returned None")
                all_chunks_success = False 
            
        except Exception as e:
            print(f"    FATAL ERROR during capture chunk {i+1}: {e}")
            import traceback
            traceback.print_exc()
            all_chunks_success = False 
        
        # 4. CRITICAL: Cleanup hooks and manager immediately
        finally:
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'remove_hooks'):
                model.transformer.remove_hooks()
                model.transformer.clear_activations()
            
            # Delete manager and config to free memory
            del hook_manager
            del hook_config
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if not all_chunks_success:
            print("    Aborting remaining chunks for this speaker due to error.")
            break # Stop processing this speaker
    
    if all_chunks_success:
        print(f"\n  All capture chunks completed. Files saved to {run_temp_dir}.")
    
    return all_chunks_success


# --- ================================================ ---
# --- FIXED: PHASE 2 - PACK
# --- ================================================ ---

def run_speaker_packing(
    speaker_id: str,
    output_dir: Path,
    temp_dir_base: Path,
    split: str,
    run_type: str = "pack"
) -> bool:
    """
    Reads temporary files and packs them into WebDataset shards.
    Uses per-timestep writing to minimize memory usage.
    
    FIX: Added better error handling and cleanup
    """
    print(f"\n--- PHASE 2: {run_type.upper()} Packing for Speaker {speaker_id} ---")
    
    run_temp_dir = temp_dir_base / speaker_id
    
    if not run_temp_dir.exists():
        print(f"  Error: Temporary directory {run_temp_dir} not found. Skipping packing.")
        return False

    success = False
    try:
        # Create per-timestep writer
        shard_writer = PerTimestepShardWriter(
            output_dir=output_dir,
            speaker_id=speaker_id,
            split=split
        )
        
        # Pack data incrementally (one timestep at a time)
        samples_written = pack_speaker_incrementally(run_temp_dir, shard_writer, speaker_id)
        
        if samples_written > 0:
            print(f"  Packing completed. Wrote {samples_written} samples total.")
            success = True
        else:
            print(f"  Warning: No samples written for speaker {speaker_id}")
        
    except Exception as e:
        print(f"  FATAL ERROR during packing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # FIX: Always cleanup temp directory
        try:
            if run_temp_dir.exists():
                shutil.rmtree(run_temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"  Warning: Could not remove temp directory: {e}")
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return success


# --- ================================================ ---
# --- FIXED: Main function
# --- ================================================ ---

def main():
    print("\n" + "="*80)
    print("EXTRACTION PIPELINE - FIXED VERSION")
    print("="*80 + "\n")
    
    # --- 1. CONFIGURE YOUR PATHS ---
    LIBRITTS_ROOT = Path("/home/ewong/links/scratch/LibriTTS")
    SUBSET_TO_PROCESS = "test-clean"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    OUTPUT_DIR = Path("./extracted_features_sharded")
    TRAIN_TARS_DIR = OUTPUT_DIR / "train_tars"
    TEST_TARS_DIR = OUTPUT_DIR / "test_tars"
    TEMP_DIR = OUTPUT_DIR / "tmp" 
    
    # --- 2. SETUP DIRECTORIES ---
    TRAIN_TARS_DIR.mkdir(parents=True, exist_ok=True)
    TEST_TARS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model
    try:
        model, vocoder = load_f5_model_and_vocoder(device=DEVICE)
    except Exception as e:
        print(f"Fatal error loading model: {e}")
        return
    
    # --- 3. DEFINE INFERENCE PARAMS ---
    inference_params = {
        'mel_spec_type': 'vocos',
        'target_rms': 0.1,
        'cross_fade_duration': 0.15,
        'nfe_step': 32,
        'cfg_strength': 2.0,
        'sway_sampling_coef': -1,
        'speed': 1.0,
        'fix_duration': None,
        'device': DEVICE
    }
    
    # --- 4. FIND ALL SPEAKERS IN THE SUBSET ---
    subset_dir = LIBRITTS_ROOT / SUBSET_TO_PROCESS
    if not subset_dir.exists():
        print(f"ERROR: Subset directory {subset_dir} does not exist!")
        return
    
    speaker_dirs = sorted([p for p in subset_dir.iterdir() if p.is_dir()])
    print(f"Found {len(speaker_dirs)} speakers in {SUBSET_TO_PROCESS}.")

    if len(speaker_dirs) == 0:
        print("No speakers found! Exiting.")
        return

    # --- 5. LOOP OVER EACH SPEAKER ---
    successful_count = 0
    failed_count = 0
    
    for i, speaker_dir in enumerate(speaker_dirs):
        # SANITY CHECK 
        if i >= 1:
            print(f"\n{'='*80}")
            print("SANITY CHECK: Stopping after one speaker.")
            print(f"Successfully processed: {successful_count}")
            print(f"Failed: {failed_count}")
            print(f"{'='*80}\n")
            break 
            
        speaker_id = speaker_dir.name
        print(f"\n{'='*80}\nProcessing Speaker {i+1}/{len(speaker_dirs)}: {speaker_id}\n{'='*80}")

        # Find files
        all_wavs = sorted(list(speaker_dir.rglob("*.wav")))
        if len(all_wavs) < 2:
            print(f"  Skipping speaker {speaker_id}: not enough WAV files (need at least 2)")
            failed_count += 1
            continue
            
        train_wav_path = all_wavs[0]
        train_text_path = train_wav_path.with_suffix(".normalized.txt")
        test_wav_path = all_wavs[1]
        test_text_path = test_wav_path.with_suffix(".normalized.txt")

        if not (train_text_path.exists() and test_text_path.exists()):
            print(f"  Skipping speaker {speaker_id}: missing text files")
            failed_count += 1
            continue
        
        try:
            train_text_prompt = train_text_path.read_text().strip()
            test_text_prompt = test_text_path.read_text().strip()
        except Exception as e:
            print(f"  Skipping speaker {speaker_id}: error reading text files: {e}")
            failed_count += 1
            continue
        
        speaker_success = True
        
        # =========================================================
        # TRAIN DATA: CAPTURE THEN PACK (PER-TIMESTEP)
        # =========================================================
        
        print(f"\n{'─'*80}")
        print(f"TRAIN DATA for Speaker {speaker_id}")
        print(f"{'─'*80}")
        
        # PHASE 1: CAPTURE
        capture_success = run_speaker_capture(
            model=model, vocoder=vocoder, speaker_id=speaker_id, 
            wav_path=train_wav_path, text_prompt=train_text_prompt,
            temp_dir_base=TEMP_DIR / "train", inference_params=inference_params,
            device=DEVICE, run_type="TRAIN"
        )
        
        # FIX: Always cleanup after capture
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # PHASE 2: PACK (PER-TIMESTEP)
        if capture_success:
            pack_success = run_speaker_packing(
                speaker_id=speaker_id,
                output_dir=TRAIN_TARS_DIR,
                temp_dir_base=TEMP_DIR / "train",
                split="train",
                run_type="TRAIN"
            )
            speaker_success = speaker_success and pack_success
        else:
            speaker_success = False

        # =========================================================
        # TEST DATA: CAPTURE THEN PACK (PER-TIMESTEP)
        # =========================================================
        
        print(f"\n{'─'*80}")
        print(f"TEST DATA for Speaker {speaker_id}")
        print(f"{'─'*80}")
        
        # PHASE 1: CAPTURE
        capture_success = run_speaker_capture(
            model=model, vocoder=vocoder, speaker_id=speaker_id, 
            wav_path=test_wav_path, text_prompt=test_text_prompt,
            temp_dir_base=TEMP_DIR / "test", inference_params=inference_params,
            device=DEVICE, run_type="TEST"
        )

        # FIX: Always cleanup after capture
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # PHASE 2: PACK (PER-TIMESTEP)
        if capture_success:
            pack_success = run_speaker_packing(
                speaker_id=speaker_id,
                output_dir=TEST_TARS_DIR,
                temp_dir_base=TEMP_DIR / "test",
                split="test",
                run_type="TEST"
            )
            speaker_success = speaker_success and pack_success
        else:
            speaker_success = False
        
        # Update counters
        if speaker_success:
            successful_count += 1
        else:
            failed_count += 1
        
        # Final garbage collection after speaker
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"✓ All shards saved to {OUTPUT_DIR}")
    print(f"  - Train shards: {TRAIN_TARS_DIR}")
    print(f"  - Test shards: {TEST_TARS_DIR}")
    print(f"✓ Successfully processed: {successful_count} speakers")
    if failed_count > 0:
        print(f"✗ Failed: {failed_count} speakers")
    
    # FIX: Always cleanup temp directory at end
    if TEMP_DIR.exists():
        try:
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
            print("✓ Cleaned up temporary directory.")
        except Exception as e:
            print(f"Warning: Could not fully clean temp directory: {e}")

if __name__ == "__main__":
    main()