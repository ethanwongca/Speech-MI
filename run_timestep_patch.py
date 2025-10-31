#!/usr/bin/env python3
"""
F5-TTS DiT Timestep-Specific Patch Experiment
Tests patching a specific submodule at a specific CFM timestep.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import soundfile as sf

# --- Add parent directories to path ---
# This assumes the script is saved in the same directory as the ablation script
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

# --- Import from your modified CFM ---
try:
    from f5_tts.model.cfm import HookConfig, HookManager
except ImportError:
    print("ERROR: Could not import HookConfig/HookManager from f5_tts.model.cfm.")
    print("Please ensure your modified cfm.py is in the correct f5_tts/model/ directory.")
    sys.exit(1)

# --- Import F5-TTS components (assuming paths are set) ---
from f5_tts.infer.utils_infer import (
    load_model, load_vocoder, preprocess_ref_audio_text, infer_process
)
from omegaconf import OmegaConf
from hydra.utils import get_class
from importlib.resources import files

# =======================================
# HOOK TIME FILTERS (Copied from your script)
# =======================================
def time_filter_specific_steps(steps: list[int]):
    """Capture only specific timestep indices"""
    steps_set = set(steps)
    return lambda t, step: step in steps_set

# =======================================
# HELPER FUNCTIONS
# =======================================

def print_tensor_stats(name: str, tensor: torch.Tensor):
    """Prints summary statistics for a given tensor."""
    if tensor is None:
        print(f"  {name}: ERROR - Tensor is None")
        return
    print(f"  {name}:")
    print(f"    Shape: {tuple(tensor.shape)}")
    print(f"    Mean:  {tensor.mean().item():.6f}")
    print(f"    Std:   {tensor.std().item():.6f}")
    print(f"    Min:   {tensor.min().item():.6f}")
    print(f"    Max:   {tensor.max().item():.6f}")

def run_inference_with_hooks(
    model, 
    vocoder, 
    ref_audio, 
    ref_text, 
    gen_text, 
    experiment_name="hook_run",
    hook_config_params=None,
    **inference_kwargs
):
    """
    Runs inference with a specific HookConfig.
    
    This is a simplified version of your 'run_ablation_experiment'.
    It's designed to not save files, but to return the in-memory
    activations for direct comparison.
    """
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*70}")
    
    # Access the DiT model
    dit_model = (model.ema_model if hasattr(model, 'ema_model') else model).transformer

    # --- Build HookManager ---
    if hook_config_params is None:
        hook_config_params = {}
        
    # **Crucial settings for this test**:
    # - We don't save to disk (save_per_timestep=False)
    # - We *must* keep activations in memory (clear_activations_per_step=False)
    hook_cfg = HookConfig(
        capture_layers=hook_config_params.get('capture_layers', {}),
        patches=hook_config_params.get('patches', {}),
        time_filter=hook_config_params.get('time_filter'),
        save_per_timestep=False,
        clear_activations_per_step=False # <-- IMPORTANT!
    )
    
    hook_manager = HookManager(hook_cfg)
    inference_kwargs['hook_manager'] = hook_manager
    
    print(f"Running inference with {len(hook_cfg.capture_layers)} extraction hooks...")
    if hook_cfg.patches:
        print(f"Applying patches at steps: {list(hook_cfg.patches.keys())}")
    
    try:
        audio_segment, sample_rate, spectrogram = infer_process(
            ref_audio, ref_text, gen_text, model, vocoder, **inference_kwargs
        )
        
        # After inference, activations are in dit_model.extracted_activations
        captured_activations = dict(dit_model.extracted_activations)
        
        print(f"\nResult: SUCCESS")
        print(f"  Captured {len(captured_activations)} activations.")
        
        return {
            'audio': audio_segment,
            'sample_rate': sample_rate,
            'spectrogram': spectrogram,
            'captured_activations': captured_activations, # Return the dict
            'success': True
        }
        
    except Exception as e:
        print(f"\nResult: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return {
            'audio': None,
            'sample_rate': None,
            'spectrogram': None,
            'captured_activations': {},
            'success': False
        }

# =======================================
# MAIN EXPERIMENT RUNNER
# =======================================

def main():
    """Main timestep patch experiment runner."""
    
    print("\n" + "="*70)
    print("F5-TTS DiT TIMESTEP PATCH EXPERIMENT")
    print("="*70 + "\n")
    
    # === MODIFICATION: ADDED DETERMINISM FIX ===
    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # ===========================================
    
    # --- 1. Configuration ---
    model_name = "F5TTS_v1_Base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # LOCAL PATHS - update these
    LOCAL_MODEL_PATH = "/home/ewong/links/scratch/checkpoints/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    LOCAL_VOCODER_PATH = "/home/ewong/links/scratch/checkpoints/vocos-mel-24khz"
    
    # --- Experiment Parameters ---
    TARGET_STEP = 10  # The ODE step index to patch
    TARGET_MODULE_NAME = "block_5_attn_query"
    TARGET_MODULE_PATH = "transformer_blocks.5.attn.to_q"
    PATCH_VALUE = 0.0 # The value to patch with (e.g., zero out)
    
    OUTPUT_DIR = Path("./timestep_patch_results")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Experiment Setup ---")
    print(f"Target Step:   {TARGET_STEP}")
    print(f"Target Module: {TARGET_MODULE_PATH}")
    print(f"Patch Value:   {PATCH_VALUE}")
    print(f"Output Dir:    {OUTPUT_DIR}")
    print("------------------------\n")
    
    # --- 2. Load Model & Vocoder (Copied from your script) ---
    if not Path(LOCAL_MODEL_PATH).exists():
        print(f"ERROR: Model checkpoint not found at {LOCAL_MODEL_PATH}")
        return
    if not Path(LOCAL_VOCODER_PATH).exists():
        print(f"ERROR: Vocoder not found at {LOCAL_VOCODER_PATH}")
        return
    
    print(f"Device: {device}")
    
    print("Loading vocoder...")
    vocoder = load_vocoder(
        vocoder_name="vocos", is_local=True,
        local_path=LOCAL_VOCODER_PATH, device=device
    )
    print("Loading F5-TTS model...")
    model_cfg = OmegaConf.load(
        str(files("f5_tts").joinpath(f"configs/{model_name}.yaml"))
    )
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arch = model_cfg.model.arch
    
    model = load_model(
        model_cls, model_arch, LOCAL_MODEL_PATH,
        mel_spec_type="vocos", vocab_file="", device=device
    )
    print("Model and Vocoder loaded.\n")
    
    # --- 3. Load Data (Copied from your script) ---
    ref_audio = "Original Speech and Text/84_121123_000008_000000.wav"
    ref_text_file = "Original Speech and Text/84_121123_000008_000000.normalized.txt"
    
    if not Path(ref_audio).exists() or not Path(ref_text_file).exists():
        print(f"ERROR: Reference files not found")
        return
        
    with open(ref_text_file, 'r', encoding='utf-8') as f:
        ref_text = f.read().strip()
    
    gen_text = "This is a test of the timestep patching system."
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text)
    
    # --- 4. Define Inference Parameters ---
    inference_params = {
        'mel_spec_type': 'vocos', 'target_rms': 0.1,
        'cross_fade_duration': 0.15, 'nfe_step': 32, # Note: your original script had 'nfe_step', but infer_process might expect 'n_fe_step'
        'cfg_strength': 2.0, 'sway_sampling_coef': -1,
        'speed': 1.0, 'fix_duration': None, 'device': device,
        'seed': 42
    }
    
    # --- 5. RUN 1: BASELINE (Extract Only) ---
    
    baseline_hook_params = {
        'capture_layers': {
            TARGET_MODULE_NAME: TARGET_MODULE_PATH
        },
        'time_filter': time_filter_specific_steps([TARGET_STEP]),
        'patches': {} # No patches
    }
    
    baseline_result = run_inference_with_hooks(
        model, vocoder, ref_audio, ref_text, gen_text,
        experiment_name="01_Baseline_Extract",
        hook_config_params=baseline_hook_params,
        **inference_params
    )
    
    if not baseline_result['success']:
        print("Baseline run failed. Aborting.")
        return

    # Save baseline audio
    sf.write(
        OUTPUT_DIR / "01_baseline_audio.wav", 
        baseline_result['audio'], 
        baseline_result['sample_rate']
    )
    
    # --- 6. RUN 2: PATCHED (Patch + Extract) ---
    
    # This is the new patch format: {step_index: [patch_list]}
    patch_config = {
        TARGET_STEP: [
            {
                'path': TARGET_MODULE_PATH,
                'kwargs': {
                    'patch_vec': PATCH_VALUE,
                    # --- NEW: Use the patch hook to extract ---
                    'extract_name_before': f"{TARGET_MODULE_NAME}_before_patch",
                    'extract_name_after':  f"{TARGET_MODULE_NAME}_after_patch"
                }
            }
        ]
    }
    
    patched_hook_params = {
        'capture_layers': {}, 
        'time_filter': time_filter_specific_steps([TARGET_STEP]),
        'patches': patch_config 
    }
    
    patched_result = run_inference_with_hooks(
        model, vocoder, ref_audio, ref_text, gen_text,
        experiment_name="02_Patched_Extract",
        hook_config_params=patched_hook_params,
        **inference_params
    )
    
    if not patched_result['success']:
        print("Patched run failed. Aborting.")
        return

    # Save patched audio
    sf.write(
        OUTPUT_DIR / "02_patched_audio.wav", 
        patched_result['audio'], 
        patched_result['sample_rate']
    )

    # === MODIFICATION: ADDED RUN 3 ===
    # --- 7. RUN 3: POST-PATCH BASELINE (Extract Only) ---
    # This is to confirm that running another baseline after the
    # patch run still produces the same deterministic result.
    
    print("\n--- Running Post-Patch Baseline to check determinism ---")
    
    # We use the *exact same* hook_params and inference_params as Run 1
    post_baseline_result = run_inference_with_hooks(
        model, vocoder, ref_audio, ref_text, gen_text,
        experiment_name="03_Post_Patch_Baseline",
        hook_config_params=baseline_hook_params, # Re-use params from Run 1
        **inference_params
    )
    
    if not post_baseline_result['success']:
        print("Post-Patch Baseline run failed. Aborting.")
        return

    # Save post-baseline audio
    sf.write(
        OUTPUT_DIR / "03_post_baseline_audio.wav", 
        post_baseline_result['audio'], 
        post_baseline_result['sample_rate']
    )
    # ==================================
    
    # --- 8. VERIFICATION (Compare Activations) --- (Was section 7)
    
    print("\n" + "="*70)
    print(f"VERIFICATION: Comparing Activations for Step {TARGET_STEP}")
    print("="*70)
    
    try:
        # 1. Get Baseline tensor from Run 1
        baseline_tensor = baseline_result['captured_activations']\
                                       [TARGET_MODULE_NAME]['tensor']
        
        # 2. Get 'before' and 'after' tensors from Run 2
        before_patch_tensor = patched_result['captured_activations']\
                                          [f"{TARGET_MODULE_NAME}_before_patch"]['tensor']
        
        after_patch_tensor = patched_result['captured_activations']\
                                         [f"{TARGET_MODULE_NAME}_after_patch"]['tensor']
        
        # 3. Get Post-Baseline tensor from Run 3
        post_baseline_tensor = post_baseline_result['captured_activations']\
                                                  [TARGET_MODULE_NAME]['tensor']
                                          
        print(f"Stats for: {TARGET_MODULE_PATH}\n")
        print_tensor_stats("Run 1: Baseline", baseline_tensor)
        print_tensor_stats("Run 2: Before Patch (from patch hook)", before_patch_tensor)
        print_tensor_stats("Run 2: After Patch (from patch hook)", after_patch_tensor)
        # === MODIFICATION: ADDED RUN 3 STATS ===
        print_tensor_stats("Run 3: Post-Patch Baseline", post_baseline_tensor)
        
        # --- MODIFICATION: UPDATED FINAL CHECKS ---
        print("\n--- Verification Check ---")
        
        # Check 1: Baseline and "Before Patch" should be identical
        is_baseline_match = torch.allclose(baseline_tensor, before_patch_tensor, atol=1e-5)
        print(f"Run 1 matches Run 2 'Before Patch': {is_baseline_match}")

        # Check 2: "After Patch" should match the patch value
        if isinstance(PATCH_VALUE, (int, float)):
            patch_val_tensor = torch.full_like(after_patch_tensor, PATCH_VALUE)
        else:
            patch_val_tensor = PATCH_VALUE.to(after_patch_tensor.device)

        is_patch_correct = torch.allclose(after_patch_tensor, patch_val_tensor, atol=1e-5)
        print(f"Patch applied successfully ('After Patch' == 0.0): {is_patch_correct}")
            
        # Check 3: Run 1 and Run 3 should be identical
        is_run1_run3_match = torch.allclose(baseline_tensor, post_baseline_tensor, atol=1e-5)
        print(f"Run 1 matches Run 3 (proves determinism): {is_run1_run3_match}")
            
    except KeyError as e:
        print(f"ERROR: Could not find activation in results. Missing key: {e}")
    except Exception as e:
        print(f"ERROR during verification: {e}")

    print("\n" + "="*70)
    print("TIMESTEP PATCH EXPERIMENT COMPLETE")
    print(f"Results audio saved to: {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()