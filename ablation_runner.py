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
    
    Args:
        model: The loaded F5-TTS model with DiT backbone
        vocoder: Loaded vocoder
        ref_audio, ref_text, gen_text: Input data
        experiment_name: Name for this experiment
        ablation_config: Dict with hook configuration
            Example: {
                'hook_name': {
                    'type': 'zero_out' | 'constant' | 'extract',
                    'path': 'transformer_blocks.0',
                    'value': 0.0 (for constant type),
                    'batch_index': None or int,
                    'feature_index': None or int
                }
            }
    """
    from f5_tts.infer.utils_infer import infer_process
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*70}")
    
    # Access the DiT model (it's nested deeper in F5-TTS structure)
    # F5-TTS structure: EMA -> CFM -> DiT (transformer)
    dit_model = None
    if hasattr(model, 'ema_model'):
        # model is EMA wrapper
        cfm_model = model.ema_model
        if hasattr(cfm_model, 'transformer'):
            dit_model = cfm_model.transformer  # This is the actual DiT
        else:
            dit_model = cfm_model
    elif hasattr(model, 'transformer'):
        dit_model = model.transformer
    else:
        dit_model = model
    
    print(f"DiT model type: {type(dit_model).__name__}")
    
    # Clear any existing hooks
    if hasattr(dit_model, 'remove_hooks'):
        dit_model.remove_hooks()
        dit_model.clear_activations()
        print("Cleared existing hooks")
    
    # Apply ablation hooks if specified
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
                    batch_index=hook_config.get('batch_index'),
                    feature_index=hook_config.get('feature_index'),
                    output_index=hook_config.get('output_index', 0),
                    patch_vec=value
                )
                print(f"  - {hook_name}: Patch '{path}' with value={type(value)}")
                
            elif hook_type == 'extract':
                dit_model.add_extraction_hook(
                    submodule_path=path,
                    hook_name=hook_name,
                    output_index=hook_config.get('output_index', 0)
                )
                print(f"  - {hook_name}: Extract from '{path}'")
    
    # Run inference
    try:
        audio_segment, sample_rate, spectrogram, timeste_activations = infer_process(
            ref_audio, ref_text, gen_text, model, vocoder, **inference_kwargs
        )
        
        # Collect activation statistics if hooks were used
        activation_stats = {}
        hook_stats = {}
        
        # <<< MODIFIED: Also capture the raw tensors for re-insertion experiments
        captured_tensors = {}
        if hasattr(dit_model, 'extracted_activations'):
            captured_tensors = dit_model.extracted_activations
            for name, data in captured_tensors.items():
                tensor = data['tensor']
                activation_stats[name] = {
                    'shape': list(data['shape']),
                    'mean': float(tensor.mean()),
                    'std': float(tensor.std()),
                    'min': float(tensor.min()),
                    'max': float(tensor.max()),
                    'call_count': data.get('call_count', 0)
                }
        
        if hasattr(dit_model, 'get_hook_stats'):
            hook_stats = dit_model.get_hook_stats()
        
        print(f"\nResult: SUCCESS")
        print(f"  Audio length: {len(audio_segment)/sample_rate:.2f}s")
        print(f"  Sample rate: {sample_rate}")
        if activation_stats:
            print(f"  Captured {len(activation_stats)} activations")
        
        return {
            'audio': audio_segment,
            'sample_rate': sample_rate,
            'spectrogram': spectrogram,
            'activation_stats': activation_stats,
            'hook_stats': hook_stats,
            'captured_tensors': captured_tensors, # <<< MODIFIED: Return the raw tensors
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
            'spectrogram': None,
            'activation_stats': {},
            'hook_stats': {},
            'captured_tensors': {}, # <<< MODIFIED: Return empty dict on failure
            'success': False,
            'error': str(e)
        }
    finally:
        # Clean up hooks
        if hasattr(dit_model, 'remove_hooks'):
            dit_model.remove_hooks()

def define_ablation_experiments():
    """
    Define ablation study configurations for DiT.
    Returns dict of experiment_name -> ablation_config
    """
    experiments = {
        # 1. Baseline - no modifications
        "01_baseline": None,
        
        # 2. Zero out entire transformer blocks
        "02_zero_block0": {
            "zero_b0": {
                "type": "zero_out",
                "path": "transformer_blocks.0",
                "value": 0.0
            }
        },
        
        "03_zero_block_last": {
            "zero_blast": {
                "type": "zero_out",
                "path": "transformer_blocks.23",  # <<< MODIFIED: Corrected to the actual last block
                "value": 0.0
            }
        },
        
        # 3. Zero out input embeddings
        "04_zero_input_embed": {
            "zero_input": {
                "type": "zero_out",
                "path": "input_embed",
                "value": 0.0
            }
        },
        
        # 4. Zero out text embeddings
        "05_zero_text_embed": {
            "zero_text": {
                "type": "zero_out",
                "path": "text_embed",
                "value": 0.0
            }
        },
        
        # 5. Zero out specific feature dimension
        "06_zero_feature_dim10": {
            "zero_feat10": {
                "type": "zero_out",
                "path": "transformer_blocks.2",
                "value": 0.0,
                "feature_index": 10
            }
        },
        
        # 6. Constant imputation (set to mean-like value)
        "07_constant_block0": {
            "const_b0": {
                "type": "constant",
                "path": "transformer_blocks.0",
                "value": 0.01  # Small non-zero constant
            }
        },
        
        # 7. Constant imputation on projection
        "08_constant_input_proj": {
            "const_proj": {
                "type": "constant",
                "path": "input_embed.proj",
                "value": 0.5
            }
        },
        
        # 8. Zero out time embedding
        "09_zero_time_embed": {
            "zero_time": {
                "type": "zero_out",
                "path": "time_embed",
                "value": 0.0
            }
        },
        
        # 9. Multiple ablations combined
        "10_multi_zero_blocks": {
            "zero_b0": {
                "type": "zero_out",
                "path": "transformer_blocks.0",
                "value": 0.0
            },
            "zero_b1": {
                "type": "zero_out",
                "path": "transformer_blocks.1",
                "value": 0.0
            }
        },
        
        # 10. Feature extraction for analysis
        "11_extract_activations": {
            "ext_input": {
                "type": "extract",
                "path": "input_embed"
            },
            "ext_b0": {
                "type": "extract",
                "path": "transformer_blocks.0"
            },
            "ext_b_mid": {
                "type": "extract",
                "path": "transformer_blocks.11" # <<< MODIFIED: Adjusted for 24 blocks
            },
            "ext_b_last": {
                "type": "extract",
                "path": "transformer_blocks.23" # <<< MODIFIED: Adjusted for 24 blocks
            }
        }
    }
    
    return experiments

def save_results(results, output_dir, gen_text):
    """Save experiment results and audio files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(output_dir) / f"dit_ablation_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'gen_text': gen_text,
        'experiments': {}
    }
    
    # Save audio files and collect metadata
    for exp_name, result in results.items():
        exp_metadata = {
            'success': result['success'],
            'error': result['error'],
            'activation_stats': result['activation_stats'],
            'hook_stats': result['hook_stats']
        }
        
        if result['success'] and result['audio'] is not None:
            # Save audio
            audio_path = results_dir / f"{exp_name}.wav"
            sf.write(audio_path, result['audio'], result['sample_rate'])
            
            exp_metadata.update({
                'audio_file': str(audio_path.name),
                'sample_rate': result['sample_rate'],
                'audio_length_sec': len(result['audio']) / result['sample_rate'],
            })
        
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
            if result['error']:
                f.write(f"Error: {result['error']}\n")
            if result['activation_stats']:
                f.write(f"\nActivation Statistics:\n")
                for name, stats in result['activation_stats'].items():
                    f.write(f"  {name}:\n")
                    f.write(f"    Shape: {stats['shape']}\n")
                    f.write(f"    Mean: {stats['mean']:.6f}\n")
                    f.write(f"    Std: {stats['std']:.6f}\n")
    
    print(f"\nResults saved to: {results_dir}")
    return results_dir

def main():
    """Main ablation study runner."""
    
    print("\n" + "="*70)
    print("F5-TTS DiT ABLATION STUDY")
    print("="*70 + "\n")
    
    # Import F5-TTS components
    from f5_tts.infer.utils_infer import (
        load_model, load_vocoder, preprocess_ref_audio_text
    )
    from omegaconf import OmegaConf
    from hydra.utils import get_class
    from importlib.resources import files
    
    # Configuration
    model_name = "F5TTS_v1_Base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # LOCAL PATHS - update these to your local checkpoint locations
    LOCAL_MODEL_PATH = "/home/ewong/links/scratch/checkpoints/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    LOCAL_VOCODER_PATH = "/home/ewong/links/scratch/checkpoints/vocos-mel-24khz"
    
    # Check if local files exist
    if not Path(LOCAL_MODEL_PATH).exists():
        print(f"ERROR: Model checkpoint not found at {LOCAL_MODEL_PATH}")
        print("Please download using the commands provided.")
        return
    
    if not Path(LOCAL_VOCODER_PATH).exists():
        print(f"ERROR: Vocoder not found at {LOCAL_VOCODER_PATH}")
        print("Please download using the commands provided.")
        return
    
    print(f"Device: {device}")
    print(f"Loading model from: {LOCAL_MODEL_PATH}")
    print(f"Loading vocoder from: {LOCAL_VOCODER_PATH}\n")
    
    # Load vocoder
    print("Loading vocoder...")
    vocoder = load_vocoder(
        vocoder_name="vocos", 
        is_local=True,
        local_path=LOCAL_VOCODER_PATH,
        device=device
    )
    print("Vocoder loaded\n")
    
    # Load F5-TTS model
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
    print("Model loaded with DiT backbone\n")
    
    # Print DiT structure - navigate to the actual DiT
    cfm_model = model.ema_model if hasattr(model, 'ema_model') else model
    dit_model = cfm_model.transformer if hasattr(cfm_model, 'transformer') else cfm_model
    
    print(f"Model structure:")
    print(f"  Top level: {type(model).__name__}")
    print(f"  CFM level: {type(cfm_model).__name__}")
    print(f"  DiT level: {type(dit_model).__name__}")
    
    if hasattr(dit_model, 'print_available_submodules'):
        dit_model.print_available_submodules(max_depth=2)
    
    # Setup experiment data
    ref_audio = "Original Speech and Text/84_121123_000008_000000.wav"
    ref_text_file = "Original Speech and Text/84_121123_000008_000000.normalized.txt"
    
    # Check if files exist
    if not Path(ref_audio).exists() or not Path(ref_text_file).exists():
        print(f"ERROR: Reference files not found")
        print(f"  Audio: {ref_audio}")
        print(f"  Text: {ref_text_file}")
        return
    
    with open(ref_text_file, 'r', encoding='utf-8') as f:
        ref_text = f.read().strip()
    
    gen_text = "This is a test of the ablation system."
    
    print(f"Reference audio: {ref_audio}")
    print(f"Reference text: {ref_text[:50]}...")
    print(f"Generation text: {gen_text}\n")
    
    # Preprocess reference audio
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text)
    
    # Define experiments
    experiments = define_ablation_experiments()
    
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
    
    # <<< MODIFIED: STAGE 1 - CAPTURE THE BASELINE ACTIVATION ---
    print("\n" + "="*70)
    print("STAGE 1: Capturing baseline activation from 'transformer_blocks.0'")
    print("="*70)
    
    capture_config = {
        "capture_b0": {
            "type": "extract",
            "path": "transformer_blocks.0"
        }
    }
    capture_result = run_ablation_experiment(
        model=model, vocoder=vocoder, ref_audio=ref_audio, ref_text=ref_text, 
        gen_text=gen_text, experiment_name="capture_run", 
        ablation_config=capture_config, **inference_params
    )
    
    captured_tensor = capture_result.get('captured_tensors', {})\
                                    .get('capture_b0', {})\
                                    .get('tensor')

    if captured_tensor is None:
        print("ERROR: Failed to capture the activation tensor. Aborting.")
        return
        
    print(f"Successfully captured tensor with shape: {captured_tensor.shape}")

    # <<< MODIFIED: STAGE 2 - ADD THE 'RE-INSERT' EXPERIMENT ---
    print("\n" + "="*70)
    print("STAGE 2: Defining and running ablation experiments")
    print("="*70)

    experiments["12_reinsert_block0_identity"] = {
        "reinsert_b0": {
            "type": "constant",
            "path": "transformer_blocks.0",
            "value": captured_tensor
        }
    }
    experiments["13_reinsert_block0_to_block4"] = {
        "reinsert_b0_to_b4": {
            "type": "constant",
            "path": "transformer_blocks.4", # Patch a different block
            "value": captured_tensor
        }
    }
    
    # <<< MODIFIED: STAGE 3 - RUN ALL EXPERIMENTS ---
    results = {}
    # We can save the capture run's result if we want
    results["00_capture_run"] = capture_result 

    for exp_name, ablation_config in experiments.items():
        result = run_ablation_experiment(
            model=model,
            vocoder=vocoder,
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=gen_text,
            experiment_name=exp_name,
            ablation_config=ablation_config,
            **inference_params
        )
        results[exp_name] = result
    
    # Save results
    output_dir = "ablation_results"
    results_dir = save_results(results, output_dir, gen_text)
    
    # Print summary
    print("\n" + "="*70)
    print("ABLATION STUDY COMPLETE")
    print("="*70)
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    print(f"\nSuccessful experiments: {successful}/{total}")
    print(f"Results directory: {results_dir}")
    
    print("\nNext steps:")
    print("  1. Listen to audio files in the results directory")
    print("  2. Review experiment_metadata.json for activation statistics")
    print("  3. Compare spectrograms across experiments")
    print("  4. Identify which components are critical for quality")

if __name__ == "__main__":
    main()