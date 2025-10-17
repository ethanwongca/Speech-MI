#!/usr/bin/env python3
"""
F5-TTS DiT Ablation Study Runner (MODIFIED FOR NEW CFM/UTILS)
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
    from f5_tts.infer.utils_infer import infer_process
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*70}")
    
    # Access the DiT model
    cfm_model = model.ema_model if hasattr(model, 'ema_model') else model
    dit_model = cfm_model.transformer if hasattr(cfm_model, 'transformer') else cfm_model
    
    print(f"DiT model type: {type(dit_model).__name__}")
    
    # --- MODIFIED HOOKING LOGIC ---
    # We separate patch hooks (applied here) from extraction hooks (passed to the model)
    patch_hooks_config = {}
    extraction_hooks_config = {}

    if ablation_config:
        for hook_name, config in ablation_config.items():
            if config['type'] == 'extract':
                extraction_hooks_config[hook_name] = config
            else:
                patch_hooks_config[hook_name] = config

    # Apply patch hooks directly before inference
    if patch_hooks_config:
        print(f"\nApplying {len(patch_hooks_config)} patch hook(s):")
        for hook_name, hook_config in patch_hooks_config.items():
            path = hook_config['path']
            value = hook_config.get('value', 0.0)
            dit_model.add_patch_hook(
                submodule_path=path,
                hook_name=hook_name,
                patch_vec=value,
                # Add other patch params if you use them
                batch_index=hook_config.get('batch_index'),
                feature_index=hook_config.get('feature_index')
            )
            print(f"  - {hook_name}: Patching '{path}'")
    
    # Run inference, passing the extraction config to the modified CFM
    try:
        # MODIFICATION: Unpack all FOUR return values
        audio_segment, sample_rate, spectrogram, all_timestep_activations = infer_process(
            ref_audio, ref_text, gen_text, model, vocoder,
            capture_hooks_config=extraction_hooks_config if extraction_hooks_config else None,
            **inference_kwargs
        )
        
        activation_stats = {}
        captured_tensors = {} # For single-shot extractions like the original script

        # MODIFICATION: Process the new timestep dictionary
        if all_timestep_activations:
            # For compatibility, we'll get the activation from the LAST timestep
            last_timestep_key = list(all_timestep_activations.keys())[-1]
            last_step_activations = all_timestep_activations[last_timestep_key]

            for hook_name, tensor in last_step_activations.items():
                # Re-create the structure the old script expects
                captured_tensors[hook_name] = {'tensor': tensor.to(torch.float32)}
                activation_stats[hook_name] = {
                    'shape': list(tensor.shape),
                    'mean': float(tensor.mean()),
                    'std': float(tensor.std()),
                    'min': float(tensor.min()),
                    'max': float(tensor.max()),
                }
        
        print(f"\nResult: SUCCESS")
        print(f"  Audio length: {len(audio_segment)/sample_rate:.2f}s")
        if activation_stats:
            print(f"  Captured {len(activation_stats)} activations from the final timestep.")
        
        return {
            'audio': audio_segment,
            'sample_rate': sample_rate,
            'spectrogram': spectrogram,
            'activation_stats': activation_stats,
            'captured_tensors': captured_tensors,
            'all_timestep_activations': all_timestep_activations, # The full data
            'success': True,
            'error': None
        }
        
    except Exception as e:
        print(f"\nResult: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return {
            'audio': None, 'sample_rate': None, 'spectrogram': None,
            'activation_stats': {}, 'captured_tensors': {},
            'all_timestep_activations': {},
            'success': False, 'error': str(e)
        }
    finally:
        # Clean up any patch hooks we applied
        if patch_hooks_config and hasattr(dit_model, 'remove_hooks'):
            dit_model.remove_hooks()
            print("Cleaned up patch hooks.")

def define_ablation_experiments():
    """Define ablation study configurations for DiT."""
    experiments = {
        "01_baseline": None,
        "02_zero_block0": {"zero_b0": {"type": "zero_out", "path": "transformer_blocks.0"}},
        "03_zero_block_last": {"zero_blast": {"type": "zero_out", "path": "transformer_blocks.21"}},
        "04_zero_input_embed": {"zero_input": {"type": "zero_out", "path": "input_embed"}},
        "05_zero_text_embed": {"zero_text": {"type": "zero_out", "path": "text_embed"}},
        "07_constant_block0": {"const_b0": {"type": "constant", "path": "transformer_blocks.0", "value": 0.01}},
        "11_extract_activations": {
            "ext_input": {"type": "extract", "path": "input_embed"},
            "ext_b0": {"type": "extract", "path": "transformer_blocks.0"},
            "ext_b_mid": {"type": "extract", "path": "transformer_blocks.11"},
            "ext_b_last": {"type": "extract", "path": "transformer_blocks.21"}
        }
    }
    return experiments

def save_results(results, output_dir, gen_text):
    """Save experiment results and audio files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(output_dir) / f"dit_ablation_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {'timestamp': timestamp, 'gen_text': gen_text, 'experiments': {}}
    
    for exp_name, result in results.items():
        exp_metadata = {
            'success': result['success'],
            'error': result['error'],
            'activation_stats': result.get('activation_stats', {})
        }
        
        if result['success'] and result['audio'] is not None:
            audio_path = results_dir / f"{exp_name}.wav"
            sf.write(audio_path, result['audio'], result['sample_rate'])
            exp_metadata.update({
                'audio_file': str(audio_path.name),
                'sample_rate': result['sample_rate']
            })
        
        # Save full timestep data for extraction experiments separately
        if result.get('all_timestep_activations'):
            ts_path = results_dir / f"{exp_name}_timesteps.pt"
            torch.save(result['all_timestep_activations'], ts_path)
            exp_metadata['timestep_data_file'] = str(ts_path.name)

        metadata['experiments'][exp_name] = exp_metadata
    
    metadata_path = results_dir / "experiment_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")
    return results_dir

def main():
    """Main ablation study runner."""
    print("\n" + "="*70 + "\nF5-TTS DiT ABLATION STUDY\n" + "="*70 + "\n")
    
    from f5_tts.infer.utils_infer import load_model, load_vocoder, preprocess_ref_audio_text
    from omegaconf import OmegaConf
    from hydra.utils import get_class
    from importlib.resources import files
    
    model_name = "F5TTS_v1_Base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    LOCAL_MODEL_PATH = "/home/ewong/links/scratch/checkpoints/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    LOCAL_VOCODER_PATH = "/home/ewong/links/scratch/checkpoints/vocos-mel-24khz"
    
    vocoder = load_vocoder(is_local=True, local_path=LOCAL_VOCODER_PATH, device=device)
    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model_name}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arch = model_cfg.model.arch
    model = load_model(model_cls, model_arch, LOCAL_MODEL_PATH, mel_spec_type="vocos", device=device)
    
    ref_audio_path = "Orignal Speech and Text/84_121123_000008_000000.wav"
    ref_text_path = "Original Speech and Text/84_121123_000008_000000.normalized.txt"
    with open(ref_text_path, 'r', encoding='utf-8') as f:
        ref_text_content = f.read().strip()
    
    gen_text = "This is a test of the ablation system."
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text_content)
    
    experiments = define_ablation_experiments()
    
    inference_params = {
        'nfe_step': 32, 'cfg_strength': 2.0, 'device': device
    }
    
    # --- STAGE 1: CAPTURE BASELINE ACTIVATION ---
    print("\n" + "="*70 + "\nSTAGE 1: Capturing baseline activation for re-insertion\n" + "="*70)
    capture_config = {"capture_b0": {"type": "extract", "path": "transformer_blocks.0"}}
    capture_result = run_ablation_experiment(
        model, vocoder, ref_audio, ref_text, gen_text, 
        experiment_name="capture_run", 
        ablation_config=capture_config, **inference_params
    )
    
    captured_tensor = capture_result.get('captured_tensors', {}).get('capture_b0', {}).get('tensor')
    if captured_tensor is None:
        print("ERROR: Failed to capture the activation tensor. Aborting.")
        return
    print(f"Successfully captured tensor with shape: {captured_tensor.shape}")

    # --- STAGE 2: ADD RE-INSERT EXPERIMENTS ---
    experiments["12_reinsert_block0_identity"] = {"reinsert_b0": {"type": "constant", "path": "transformer_blocks.0", "value": captured_tensor}}
    experiments["13_reinsert_block0_to_block4"] = {"reinsert_b0_to_b4": {"type": "constant", "path": "transformer_blocks.4", "value": captured_tensor}}
    
    # --- STAGE 3: RUN ALL EXPERIMENTS ---
    print("\n" + "="*70 + "\nSTAGE 2: Running all ablation experiments\n" + "="*70)
    results = {"00_capture_run": capture_result}

    for exp_name, ablation_config in experiments.items():
        result = run_ablation_experiment(
            model, vocoder, ref_audio, ref_text, gen_text,
            experiment_name=exp_name,
            ablation_config=ablation_config,
            **inference_params
        )
        results[exp_name] = result
    
    # --- STAGE 4: SAVE RESULTS ---
    results_dir = save_results(results, "ablation_results", gen_text)
    
    print("\n" + "="*70 + "\nABLATION STUDY COMPLETE\n" + "="*70)
    successful = sum(1 for r in results.values() if r['success'])
    print(f"\nSuccessful experiments: {successful}/{len(results)}")
    print(f"Results directory: {results_dir}")

if __name__ == "__main__":
    main()