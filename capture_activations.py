#!/usr/bin/env python3
"""
F5-TTS DiT Full Activation Capture Runner
This script runs a single inference pass and saves the activations
from EVERY specified layer at EVERY ODE timestep to disk.
Modified to save per-submodule files to avoid OOM.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import soundfile as sf
from typing import Callable
import torch.nn as nn

# Add parent directories to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

# --- Core Imports from F5-TTS ---
from f5_tts.infer.utils_infer import (
    load_model, load_vocoder, preprocess_ref_audio_text, infer_process
)
from omegaconf import OmegaConf
from hydra.utils import get_class
from importlib.resources import files
from f5_tts.model.cfm import HookConfig, HookManager


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_capture_experiment(
    model, 
    vocoder, 
    ref_audio, 
    ref_text, 
    gen_text, 
    experiment_name="capture_run",
    hook_manager: HookManager | None = None,
    **inference_kwargs
):
    """
    Simplified runner that executes infer_process with a given HookManager.
    """
    
    print(f"\n{'='*70}")
    print(f"RUNNING CAPTURE: {experiment_name}")
    print(f"{'='*70}")

    if hook_manager:
        inference_kwargs['hook_manager'] = hook_manager
    
    try:
        audio_segment, sample_rate, spectrogram = infer_process(
            ref_audio, ref_text, gen_text, model, vocoder, **inference_kwargs
        )
        
        print(f"\nResult: SUCCESS")
        print(f"  Audio length: {len(audio_segment)/sample_rate:.2f}s")
        
        return {
            'audio': audio_segment,
            'sample_rate': sample_rate,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        print(f"\nResult: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return {
            'audio': None,
            'sample_rate': None,
            'success': False,
            'error': str(e)
        }
# Testing to see if layers are proper 
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
# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main capture runner."""
    
    print("\n" + "="*70)
    print("F5-TTS DiT FULL ACTIVATION CAPTURE (Per-Submodule Saving)")
    print("="*70 + "\n")
    
    # --- 1. Standard Model Loading ---
    model_name = "F5TTS_v1_Base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # LOCAL PATHS - update these if needed
    LOCAL_MODEL_PATH = "/home/ewong/links/scratch/checkpoints/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    LOCAL_VOCODER_PATH = "/home/ewong/links/scratch/checkpoints/vocos-mel-24khz"
    
    if not Path(LOCAL_MODEL_PATH).exists():
        print(f"ERROR: Model checkpoint not found at {LOCAL_MODEL_PATH}")
        return
    
    if not Path(LOCAL_VOCODER_PATH).exists():
        print(f"ERROR: Vocoder not found at {LOCAL_VOCODER_PATH}")
        return
    
    print(f"Device: {device}")
    
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

    # OPTIONAL CODE, CAN COMMENT OUT IF NOT DEBUGGING 
    '''
    print("\n" + "="*70)
    print("INSPECTING MODULE GROUPS (based on collapse logic):")
    
    if not hasattr(model, 'transformer'):
        print("FATAL ERROR: Model has no 'transformer' attribute!")
    else:
        all_group_names = set()
        
        # Iterate over all modules just like the hook setup will
        for module_path, module in model.transformer.named_modules():
            if not module_path:
                continue
            
            # Run the grouping logic on each module path
            group_name, _ = get_group_and_key(module_path)
            all_group_names.add(group_name)
            
        print(f"Found {len(all_group_names)} unique top-level groups:")
        
        # Print the clean, sorted list
        for group in sorted(list(all_group_names)):
            print(f"  - {group}")
            
    print("="*70 + "\n")
    print("Exiting script after group inspection.")
    sys.exit()
    '''
    # --- 2. Standard Data Setup ---
    ref_audio = "Original Speech and Text/84_121123_000008_000000.wav"
    ref_text_file = "Original Speech and Text/84_121123_000008_000000.normalized.txt"
    
    if not Path(ref_audio).exists() or not Path(ref_text_file).exists():
        print(f"ERROR: Reference files not found")
        return
    
    with open(ref_text_file, 'r', encoding='utf-8') as f:
        ref_text = f.read().strip()
    
    gen_text = "This is a test of the full capture system."
    
    print(f"Reference audio: {ref_audio}")
    print(f"Generation text: {gen_text}\n")
    
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text)
    
    # Inference parameters
    inference_params = {
        'mel_spec_type': 'vocos',
        'target_rms': 0.1,
        'cross_fade_duration': 0.15,
        'nfe_step': 32,
        'cfg_strength': 2.0,
        'sway_sampling_coef': -1,
        'speed': 1.0,
        'fix_duration': None,
        'device': device
    }
    
    # --- 3. Define the Capture Experiment (NEW "CAPTURE EVERYTHING" VERSION) ---
    experiment_name = "capture_EVERYTHING_all_steps"
    output_dir = Path("./capture_results") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Capture will be saved to: {output_dir.absolute()}")

    print("\n" + "="*70)
    print("Building full module capture list...")
    
    layers_to_capture = {}
    
    # The hook logic is attached to 'model.transformer', not 'model'.
    # We must iterate over 'model.transformer.named_modules()'
    # to get the correct paths relative to that submodule.
    
    if not hasattr(model, 'transformer'):
        print("FATAL ERROR: Model has no 'transformer' attribute!")
        print("This script assumes the DiT has a 'model.transformer' submodule.")
        return # Exit main()
        
    # Iterate over the transformer submodule, not the root model
    for module_path, module in model.transformer.named_modules():
        
        # The root module (model.transformer) has an empty path, skip it.
        if not module_path:
            continue
            
        # The hook_name and layer_path are now relative,
        # e.g., 'transformer_blocks.14.ff.ff.0.0'
        hook_name = module_path
        layers_to_capture[hook_name] = module_path

    print(f"Found {len(layers_to_capture)} total submodules in 'model.transformer' to capture.")
    print("="*70 + "\n")
    
    # 1. Create the HookConfig as requested
    capture_config = HookConfig(
        capture_all_layers=False, 
        capture_layers=layers_to_capture, 
        save_per_timestep=True,
        time_filter=lambda t, step: True,
        save_dir=output_dir,
        clear_activations_per_step=True,
        precision=torch.bfloat16,
        device=device
    )

    # 2. Create the HookManager
    hook_manager = HookManager(capture_config)
    
    # --- 4. Run the Capture Experiment ---
    print("\n" + "="*70)
    print("STARTING FULL CAPTURE EXPERIMENT")
    print("="*70)
    
    result = run_capture_experiment(
        model=model,
        vocoder=vocoder,
        ref_audio=ref_audio,
        ref_text=ref_text,
        gen_text=gen_text,
        experiment_name=experiment_name,
        hook_manager=hook_manager,
        **inference_params
    )

    # 5. Save the resulting audio
    if result['success']:
        audio_path = output_dir / f"generated_audio.wav"
        sf.write(audio_path, result['audio'], result['sample_rate'])
        print(f"Saved generated audio to {audio_path}")
    
    print("\n" + "="*70)
    print("CAPTURE SCRIPT COMPLETE")
    print(f"Results directory: {output_dir.absolute()}")
    print(f"Each timestep is in its own subdirectory (step_XXXX/)")
    print(f"Each submodule activation is a separate .pt file")
    print("="*70)


if __name__ == "__main__":
    main()