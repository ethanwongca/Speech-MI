#!/usr/bin/env python3
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
import gc  # Added for cleanup

# --- Core Imports from F5-TTS ---
from f5_tts.infer.utils_infer import (
    load_model, load_vocoder, preprocess_ref_audio_text, infer_process
)
# We need the MODIFIED HookConfig and HookManager
from f5_tts.model.cfm import HookConfig, HookManager, CFM 
from f5_tts.model.backbones.dit import DiT 

# --- ================================================ ---
# --- Helper Functions (Load model, pack shards, etc. are unchanged)
# --- ================================================ ---
# (load_f5_model_and_vocoder and pack_speaker_data_to_shards are defined here)
# ... [Original Helper Functions go here] ...
# --- ================================================ ---
# --- Shard packing function (Unchanged)
# --- ================================================ ---

def load_f5_model_and_vocoder(device="cuda"):
    """
    Loads the F5-TTS model and vocoder using your paths.
    """
    print(f"Device: {device}")
    
    # --- 1. Standard Model Loading ---
    model_name = "F5TTS_v1_Base"
    
    # Using your paths
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
# --- Shard packing function
# --- ================================================ ---

def pack_speaker_data_to_shards(temp_dir: Path, wds_writer: webdataset.ShardWriter, speaker_id: str):
    """
    Scans a temporary hook output directory and writes each
    *grouped* tensor file to its own dedicated SHARD (tar file).
    """
    metadata_path = temp_dir / "capture_metadata.json"
    if not metadata_path.exists():
        return 0

    with open(metadata_path, 'r') as f:
        meta = json.load(f)

    samples_written = 0
    for timestep_info in meta['timesteps']:
        step_index = timestep_info['step_index']
        step_dir = Path(timestep_info['directory'])
        
        for group_name in timestep_info['group_names']:
            safe_filename = f"{group_name.replace('/', '_').replace('.', '_')}.pt"
            tensor_path = step_dir / safe_filename
            
            if not tensor_path.exists():
                print(f"  Warning: Missing grouped tensor {tensor_path}")
                continue

            try:
                tensor_data_bytes = tensor_path.read_bytes()
            except Exception as e:
                print(f"  Warning: Failed to load {tensor_path}, skipping. Error: {e}")
                continue

            sample_metadata = {
                'speaker_id': speaker_id,
                'step_index': step_index,
                'layer_group': group_name,
            }
            sample_metadata_data = json.dumps(sample_metadata).encode('utf-8')
            
            shard_key = f"step_{step_index:04d}_group_{safe_filename.split('.')[0]}"
            sample_key = speaker_id
            
            sample = {
                "__key__": sample_key,
                "pth": tensor_data_bytes,
                "json": sample_metadata_data
            }
            
            wds_writer.write(sample, shard=shard_key)
            samples_written += 1
    
    return samples_written

# --- ================================================ ---
# --- PHASE 1: CAPTURE ONLY (Memory Heavy)
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
    Runs the full hook extraction and writes activations to temp files on disk.
    Does NOT involve WebDataset packing or buffering.
    """
    print(f"\n--- PHASE 1: {run_type.upper()} Capture for Speaker {speaker_id} ({wav_path.name}) ---")
    
    run_temp_dir = temp_dir_base / speaker_id
    
    # Define layers to capture (The "Capture All" strategy)
    layers_to_capture = {}
    for module_path, module in model.transformer.named_modules():
        if not module_path: 
            continue
        hook_name = module_path
        layers_to_capture[hook_name] = module_path

    # Create LOCAL HookConfig and HookManager (Memory-intensive objects)
    hook_config = HookConfig(
        capture_all_layers=False, 
        capture_layers=layers_to_capture,
        save_per_timestep=True,
        time_filter=lambda t, step: True,
        clear_activations_per_step=True,
        precision=torch.bfloat16,
        save_dir=str(run_temp_dir),
        device=device   
    )
    hook_manager = HookManager(config=hook_config)

    # Preprocess audio and text
    try:
        ref_audio, ref_text = preprocess_ref_audio_text(str(wav_path), text_prompt)
    except Exception as e:
        print(f"  Skipping speaker {speaker_id}: error preprocessing audio/text: {e}")
        return False
    
    # Run inference
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
        
        if result is not None:
            print(f"  Capture completed successfully. Files saved to {run_temp_dir}.")
        else:
            print(f"  Warning: Capture returned None")
            return False
            
    except Exception as e:
        print(f"  FATAL ERROR during capture: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

# --- ================================================ ---
# --- PHASE 2: PACK ONLY (WebDataset Heavy)
# --- ================================================ ---

def run_speaker_packing(
    speaker_id: str,
    wds_writer: webdataset.ShardWriter,
    temp_dir_base: Path,
    run_type: str = "pack"
) -> bool:
    """
    Reads temporary files and packs them into WebDataset shards.
    """
    print(f"\n--- PHASE 2: {run_type.upper()} Packing for Speaker {speaker_id} ---")
    
    run_temp_dir = temp_dir_base / speaker_id
    
    if not run_temp_dir.exists():
        print(f"  Error: Temporary directory {run_temp_dir} not found. Skipping packing.")
        return False

    try:
        # Pack data into shards
        samples_written = pack_speaker_data_to_shards(run_temp_dir, wds_writer, speaker_id)
        
        # CRITICAL FIX: Force WebDataset buffer to flush to disk immediately
        wds_writer.flush()
        
        # Clean up temporary files
        shutil.rmtree(run_temp_dir, ignore_errors=True)

        print(f"  Packing completed. Wrote {samples_written} samples. WDS buffer flushed. Temp dir cleaned.")
        return True
    
    except Exception as e:
        print(f"  FATAL ERROR during packing/flushing: {e}")
        import traceback
        traceback.print_exc()
        # Clean up temporary files even on packing failure
        shutil.rmtree(run_temp_dir, ignore_errors=True)
        return False

# --- ================================================ ---
# --- Main function (Orchestrating Capture then Pack)
# --- ================================================ ---

def main():
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
    
    # Load your actual model
    try:
        model, vocoder = load_f5_model_and_vocoder(device=DEVICE)
    except Exception as e:
        print(f"Fatal error loading model: {e}")
        return
    
    # Define the ShardWriter patterns
    train_shard_pattern = str(TRAIN_TARS_DIR / "%s.tar")
    test_shard_pattern = str(TEST_TARS_DIR / "%s.tar")
    
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
    speaker_dirs = sorted([p for p in subset_dir.iterdir() if p.is_dir()])
    print(f"Found {len(speaker_dirs)} speakers in {SUBSET_TO_PROCESS}.")

    try:
        train_writer = webdataset.ShardWriter(train_shard_pattern)
        test_writer = webdataset.ShardWriter(test_shard_pattern)

        # --- 5. LOOP OVER EACH SPEAKER ---
        for i, speaker_dir in enumerate(speaker_dirs):
            # [!! SANITY CHECK IS STILL ENABLED !!]
            if i >= 1:
                print("Stopping after one speaker for sanity check.")
                break 
                
            speaker_id = speaker_dir.name
            print(f"\n{'='*80}\nProcessing Speaker {i+1}/{len(speaker_dirs)}: {speaker_id}\n{'='*80}")

            # Find files
            all_wavs = sorted(list(speaker_dir.rglob("*.wav")))
            if len(all_wavs) < 2: continue
            train_wav_path = all_wavs[0]
            train_text_path = train_wav_path.with_suffix(".normalized.txt")
            test_wav_path = all_wavs[1]
            test_text_path = test_wav_path.with_suffix(".normalized.txt")

            if not (train_text_path.exists() and test_text_path.exists()): continue
            
            try:
                train_text_prompt = train_text_path.read_text().strip()
                test_text_prompt = test_text_path.read_text().strip()
            except Exception:
                continue
                
            
            # =========================================================
            # TRAIN DATA: CAPTURE THEN PACK
            # =========================================================
            
            # PHASE 1: CAPTURE
            success = run_speaker_capture(
                model=model, vocoder=vocoder, speaker_id=speaker_id, 
                wav_path=train_wav_path, text_prompt=train_text_prompt,
                temp_dir_base=TEMP_DIR / "train", inference_params=inference_params,
                device=DEVICE, run_type="TRAIN"
            )
            
            gc.collect()
            torch.cuda.empty_cache()
            
            # PHASE 2: PACK
            if success:
                run_speaker_packing(
                    speaker_id=speaker_id, wds_writer=train_writer, 
                    temp_dir_base=TEMP_DIR / "train", run_type="TRAIN"
                )

            # =========================================================
            # TEST DATA: CAPTURE THEN PACK
            # =========================================================
            
            # PHASE 1: CAPTURE
            success = run_speaker_capture(
                model=model, vocoder=vocoder, speaker_id=speaker_id, 
                wav_path=test_wav_path, text_prompt=test_text_prompt,
                temp_dir_base=TEMP_DIR / "test", inference_params=inference_params,
                device=DEVICE, run_type="TEST"
            )

            # garbage collectoin 
            gc.collect()
            torch.cuda.empty_cache()

            # PHASE 2: PACK
            if success:
                run_speaker_packing(
                    speaker_id=speaker_id, wds_writer=test_writer, 
                    temp_dir_base=TEMP_DIR / "test", run_type="TEST"
                )
            
            # final garbag collection
            gc.collect()
            torch.cuda.empty_cache()

    finally:
        if 'train_writer' in locals() and train_writer: train_writer.close()
        if 'test_writer' in locals() and test_writer: test_writer.close()
        print(f"\nSharded WebDataset files saved to {OUTPUT_DIR}")
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
            print("Cleaned up temporary directory.")

if __name__ == "__main__":
    main()