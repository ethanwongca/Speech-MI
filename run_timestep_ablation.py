#!/usr/bin/env python3
"""
F5-TTS DiT Specific Timestep Analysis Script (MODIFIED with Log File)
Captures and inspects activations of a defined layer at each ODE timestep.
"""

import os
import sys
import torch
from pathlib import Path
from datetime import datetime
import soundfile as sf

# Add parent directories to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

from f5_tts.infer.utils_infer import (
    load_model, load_vocoder, preprocess_ref_audio_text, infer_process
)
from omegaconf import OmegaConf
from hydra.utils import get_class
from importlib.resources import files

def main():
    """Main specific timestep analysis runner."""
    
    print("\n" + "="*70)
    print("F5-TTS DiT SPECIFIC TIMESTEP ANALYSIS")
    print("="*70 + "\n")

    # --- 1. DEFINE YOUR EXPERIMENT HERE ---
    TARGET_HOOK_NAME = "attention_block_3_output"
    TARGET_LAYER_PATH = "transformer_blocks.3.attn" 
    
    # --- 2. CONFIGURATION ---
    OUTPUT_DIR = Path("specific_timestep_results")
    LOCAL_MODEL_PATH = "/home/ewong/links/scratch/checkpoints/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    LOCAL_VOCODER_PATH = "/home/ewong/links/scratch/checkpoints/vocos-mel-24khz"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 3. SETUP (Boilerplate) ---
    print(f"Loading models on device: {device}")
    vocoder = load_vocoder(is_local=True, local_path=LOCAL_VOCODER_PATH, device=device)
    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model = load_model(model_cls, model_cfg.model.arch, LOCAL_MODEL_PATH, device=device)

    ref_audio_path = "Original Speech and Text/84_121123_000008_000000.wav"
    ref_text_path = "Original Speech and Text/84_121123_000008_000000.normalized.txt"
    with open(ref_text_path, 'r', encoding='utf-8') as f:
        ref_text_content = f.read().strip()
    gen_text = "This analysis captures specific layer activations across time."
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text_content)
    
    inference_params = {'nfe_step': 32, 'cfg_strength': 2.0}
    
    # --- 4. RUN INFERENCE WITH SPECIFIC HOOKS ---
    print("\nStarting inference process with a specific hook enabled...")
    
    hooks_to_capture = {
        TARGET_HOOK_NAME: {"path": TARGET_LAYER_PATH}
    }
    
    audio, sample_rate, _, all_timestep_activations = infer_process(
        ref_audio=ref_audio, ref_text=ref_text, gen_text=gen_text,
        model_obj=model, vocoder=vocoder, device=device, **inference_params,
        capture_hooks_config=hooks_to_capture
    )
    
    print("Inference complete.")
    
    # --- 5. SAVE AND ANALYZE RESULTS ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = OUTPUT_DIR / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if audio is not None:
        sf.write(results_dir / "generated_audio.wav", audio, sample_rate)
        print(f"\nSaved final audio to: {results_dir / 'generated_audio.wav'}")
    
    if all_timestep_activations:
        # Save main activation stats to .txt file
        activations_path = results_dir / "specific_timestep_activations.txt"
        with open(activations_path, 'w') as f:
            f.write(f"Captured Hook: '{TARGET_HOOK_NAME}' (Path: {TARGET_LAYER_PATH})\n")
            for ts_key, hooks_data in all_timestep_activations.items():
                f.write(f"\n{'='*70}\nTIMESTEP: {ts_key}\n")
                if TARGET_HOOK_NAME in hooks_data:
                    tensor = hooks_data[TARGET_HOOK_NAME]
                    f.write(f"  Shape: {list(tensor.shape)}\n  Mean: {tensor.mean():.6f}\n")
                    f.write(f"  Data Sample (first token, first 8 values): {[f'{x:.4f}' for x in tensor[0, 0, :8].tolist()]}\n")
        print(f"Saved captured activations to: {activations_path}")

        # --- MODIFIED: SAVE FULL DEBUG LOG TO A SEPARATE FILE ---
        debug_log_path = results_dir / "debug_log.txt"
        with open(debug_log_path, 'w') as f:
            f.write("Full Timestep Data Inspection\n")
            f.write("="*40 + "\n")
            f.write("Listing all captured timesteps and the hook keys found within each:\n\n")
            for ts_key, hooks_data in all_timestep_activations.items():
                available_keys = list(hooks_data.keys())
                f.write(f"  - Timestep '{ts_key}': Found keys -> {available_keys}\n")
        print(f"Saved full inspection log to: {debug_log_path}")
        # --- END MODIFICATION ---

        print("\n--- Console Analysis Summary ---")
        timestep_keys = list(all_timestep_activations.keys())
        if len(timestep_keys) > 2:
            early_ts_key = timestep_keys[1]
            late_ts_key = timestep_keys[-1]

            print(f"Comparing early step '{early_ts_key}' with late step '{late_ts_key}'.")
            
            early_data = all_timestep_activations.get(early_ts_key, {})
            early_activation = early_data.get(TARGET_HOOK_NAME)

            late_data = all_timestep_activations.get(late_ts_key, {})
            late_activation = late_data.get(TARGET_HOOK_NAME)

            if early_activation is not None and late_activation is not None:
                print(f"\n🔬 Analyzing hook: '{TARGET_HOOK_NAME}'")
                print(f"  - Early Step ({early_ts_key}) Mean Activation: {early_activation.mean():.4f}")
                print(f"  - Late Step  ({late_ts_key}) Mean Activation: {late_activation.mean():.4f}")
            else:
                if early_activation is None:
                    print(f"\nERROR: Could not find hook key '{TARGET_HOOK_NAME}' in EARLY timestep '{early_ts_key}'.")
                if late_activation is None:
                    print(f"\nERROR: Could not find hook key '{TARGET_HOOK_NAME}' in LATE timestep '{late_ts_key}'.")
        else:
            print("\nNot enough timesteps were captured to perform a comparison.")
    else:
        print("\nNo timestep activations were captured.")

if __name__ == "__main__":
    main()
