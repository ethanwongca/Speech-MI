#!/usr/bin/env python3
"""
F5-TTS DiT Ablation Study Runner
Tests DiT model with: 1) Normal, 2) Zeroed layers, 3) Constant imputation
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import soundfile as sf

# Add parent directories to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

def run_ablation_experiment(
    model,
    vocoder,
    ref_audio,
    ref_text,
    gen_text,
    experiment_name="baseline",
    ablation_config=None,
    **inference_kwargs
):
    """
    Run a single ablation experiment on DiT model.
    """
    # MODIFIED: Import the updated infer_process
    from utils_infer import infer_process

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*70}")

    # Access the DiT model
    dit_model = None
    if hasattr(model, 'ema_model'):
        cfm_model = model.ema_model
        dit_model = cfm_model.transformer if hasattr(cfm_model, 'transformer') else cfm_model
    elif hasattr(model, 'transformer'):
        dit_model = model.transformer
    else:
        dit_model = model

    print(f"DiT model type: {type(dit_model).__name__}")

    # Clear any existing hooks from previous runs
    if hasattr(dit_model, 'remove_hooks'):
        dit_model.remove_hooks()

    # Apply ablation hooks if specified (only for patching experiments)
    if ablation_config:
        print(f"\nApplying {len(ablation_config)} hook(s):")
        for hook_name, hook_config in ablation_config.items():
            hook_type = hook_config['type']
            path = hook_config['path']

            if hook_type in ['zero_out', 'constant']:
                value = hook_config.get('value', 0.0)
                dit_model.add_patch_hook(
                    submodule_path=path,
                    hook_name=hook_name,
                    patch_vec=value,
                    **{k: v for k, v in hook_config.items() if k not in ['type', 'path', 'value']}
                )
                print(f"  - {hook_name}: Patching '{path}'")

    # Run inference
    try:
        # MODIFIED: Unpack the fourth return value which contains all activations
        audio_segment, sample_rate, spectrogram, all_activations_list = infer_process(
            ref_audio, ref_text, gen_text, model, vocoder, **inference_kwargs
        )

        # Assuming a single text chunk for ablation, so we take the first item
        timestep_activations = all_activations_list[0] if all_activations_list else {}

        print(f"\nResult: SUCCESS")
        print(f"  Audio length: {len(audio_segment)/sample_rate:.2f}s")
        print(f"  Captured activations for {len(timestep_activations)} timesteps.")

        # MODIFIED: Return the rich timestep_activations dictionary
        return {
            'audio': audio_segment,
            'sample_rate': sample_rate,
            'spectrogram': spectrogram,
            'timestep_activations': timestep_activations,
            'success': True,
            'error': None
        }

    except Exception as e:
        print(f"\nResult: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return {
            'audio': None, 'sample_rate': None, 'spectrogram': None,
            'timestep_activations': {},
            'success': False, 'error': str(e)
        }
    finally:
        # Clean up hooks
        if hasattr(dit_model, 'remove_hooks'):
            dit_model.remove_hooks()

def define_ablation_experiments():
    """
    Define ablation study configurations for DiT.
    NOTE: The 'extract' type is no longer needed, as extraction now happens automatically.
    The "11_extract_all_activations" experiment is the same as baseline, but named for clarity.
    """
    experiments = {
        "01_baseline": None,
        "02_zero_block0": {"zero_b0": {"type": "zero_out", "path": "transformer_blocks.0", "value": 0.0}},
        "03_zero_block_last": {"zero_blast": {"type": "zero_out", "path": "transformer_blocks.23", "value": 0.0}},
        "04_zero_input_embed": {"zero_input": {"type": "zero_out", "path": "input_embed", "value": 0.0}},
        "05_zero_text_embed": {"zero_text": {"type": "zero_out", "path": "text_embed", "value": 0.0}},
        "06_zero_feature_dim10": {"zero_feat10": {"type": "zero_out", "path": "transformer_blocks.2", "value": 0.0, "feature_index": 10}},
        "07_constant_block0": {"const_b0": {"type": "constant", "path": "transformer_blocks.0", "value": 0.01}},
        "08_constant_input_proj": {"const_proj": {"type": "constant", "path": "input_embed.proj", "value": 0.5}},
        "09_zero_time_embed": {"zero_time": {"type": "zero_out", "path": "time_embed", "value": 0.0}},
        "10_multi_zero_blocks": {
            "zero_b0": {"type": "zero_out", "path": "transformer_blocks.0", "value": 0.0},
            "zero_b1": {"type": "zero_out", "path": "transformer_blocks.1", "value": 0.0}
        },
        # MODIFIED: This experiment will now run like a baseline but its output will be the full activation set.
        "11_extract_all_activations": None,
    }
    return experiments

def save_results(results, output_dir, gen_text):
    """Save experiment results, audio, and activation tensors."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(output_dir) / f"dit_ablation_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    metadata = {'timestamp': timestamp, 'gen_text': gen_text, 'experiments': {}}

    for exp_name, result in results.items():
        exp_metadata = {'success': result['success'], 'error': result['error']}

        if result['success'] and result['audio'] is not None:
            audio_path = results_dir / f"{exp_name}.wav"
            sf.write(audio_path, result['audio'], result['sample_rate'])
            exp_metadata['audio_file'] = str(audio_path.name)

        # MODIFIED: Save the large activation dictionary separately using torch.save
        if result.get('timestep_activations'):
            activations_path = results_dir / f"{exp_name}_activations.pt"
            torch.save(result['timestep_activations'], activations_path)
            exp_metadata['activations_file'] = str(activations_path.name)
            # Add a summary of captured data to the JSON for quick reference
            exp_metadata['activation_summary'] = {
                'timesteps_captured': len(result['timestep_activations']),
                'layers_per_timestep': len(next(iter(result['timestep_activations'].values()), {}))
            }

        metadata['experiments'][exp_name] = exp_metadata

    # Save metadata JSON
    metadata_path = results_dir / "experiment_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save summary text file
    summary_path = results_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"DiT Ablation Study Results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Generated Text: {gen_text}\n\n")
        successful = sum(1 for r in results.values() if r['success'])
        f.write(f"Experiments: {successful}/{len(results)} successful\n\n")
        for exp_name, result in results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Status: {'SUCCESS' if result['success'] else 'FAILED'}\n")
            if result.get('error'):
                f.write(f"Error: {result['error']}\n")
            if result.get('timestep_activations'):
                summary = result['timestep_activations']
                f.write(f"Activations Captured: {len(summary)} timesteps\n")

    print(f"\nResults saved to: {results_dir}")
    return results_dir

def main():
    """Main ablation study runner."""
    print("\n" + "="*70 + "\nF5-TTS DiT ABLATION STUDY\n" + "="*70 + "\n")

    # Import F5-TTS components
    from utils.utils_infer import (
        load_model, load_vocoder, preprocess_ref_audio_text
    )
    from models.dit import DiT # MODIFIED: Import your local DiT directly
    from omegaconf import OmegaConf
    from importlib.resources import files

    # --- Configuration ---
    model_name = "F5TTS_v1_Base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOCAL_MODEL_PATH = "/home/ewong/links/scratch/F5-TTS/checkpoints/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    LOCAL_VOCODER_PATH = "/home/ewong/links/scratch/F5-TTS/checkpoints/vocos-mel-24khz"
    # ... (your local paths and model config remain the same) ...

    # --- Load Model and Vocoder ---
    print("Loading vocoder...")
    vocoder = load_vocoder(vocoder_name="vocos", is_local=True, local_path=LOCAL_VOCODER_PATH, device=device)
    print("Loading F5-TTS model...")
    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model_name}.yaml")))
    model_cls = DiT
    model = load_model(model_cls, model_cfg.model.arch, LOCAL_MODEL_PATH, mel_spec_type="vocos", device=device)
    print("Model and vocoder loaded.\n")

    # --- Setup experiment data ---
    ref_audio_path = "84_121123_000008_000000.wav"
    ref_text = "Its an amazing and terrifying discovery." # Example text, or load from file
    gen_text = "This is a test of the complete per-timestep activation capture system."
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)

    # --- Define and run experiments ---
    experiments = define_ablation_experiments()
    inference_params = {
        'mel_spec_type': 'vocos', 'nfe_step': 32, 'cfg_strength': 2.0, 'device': device
    }

    # MODIFIED: Simplified main loop. No separate capture stage is needed.
    results = {}
    for exp_name, ablation_config in experiments.items():
        result = run_ablation_experiment(
            model=model, vocoder=vocoder, ref_audio=ref_audio, ref_text=ref_text,
            gen_text=gen_text, experiment_name=exp_name,
            ablation_config=ablation_config, **inference_params
        )
        results[exp_name] = result

    # --- Save results ---
    output_dir = "ablation_results"
    results_dir = save_results(results, output_dir, gen_text)

    # --- Print summary ---
    print("\n" + "="*70 + "\nABLATION STUDY COMPLETE\n" + "="*70)
    successful = sum(1 for r in results.values() if r['success'])
    print(f"\nSuccessful experiments: {successful}/{len(results)}")
    print(f"Results directory: {results_dir}")
    print("\nNext steps:")
    print("  1. Listen to the generated .wav files.")
    print(f"  2. Load and analyze the activation tensors from the .pt files (e.g., '{next(iter(results.keys()))}_activations.pt').")

if __name__ == "__main__":
    main()