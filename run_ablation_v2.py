#!/usr/bin/env python3
"""
F5-TTS DiT Ablation Study Runner (v2)

Uses the new HookedRootModule framework (f5_tts.interpretability)
for activation patching and analysis.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import soundfile as sf
from omegaconf import OmegaConf
from hydra.utils import get_class
from importlib.resources import files

# --- Imports ---
# Add parent directory to path to find f5_tts package
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

# Import reusable functions from f5_tts.infer.utils_infer
from f5_tts.infer.utils_infer import (
    load_vocoder
    # We consciously DO NOT import load_model or infer_process
)

# Import helpers from f5_tts.model.utils
from f5_tts.model.utils import (
    list_str_to_idx
    # We don't need pinyin conversion for the base model
)
from f5_tts.data.audio import load_wav
from f5_tts.model.modules import MelSpec

# Import our new interpretability framework
from f5_tts.interpretability.hook_utils import make_patch_hook
from f5_tts.interpretability.cfm_hooked import CFMHooked
from f5_tts.interpretability.dit_hooked import HookedDiT
from f5_tts.interpretability.activation_cache import ActivationCache
# --- End Imports ---


def convert_state_dict_keys(original_state_dict):
    """
    Renames keys from the original DiT checkpoint to match our new
    HookedDiT and HookedModules (e.g., unrolls nn.Sequential).
    """
    new_state_dict = {}
    print("Converting state dict keys for hooked model...")
    
    for old_key, value in original_state_dict.items():
        new_key = old_key
        
        # --- Handle nn.Sequential unrolling ---
        
        # DiTBlock -> FeedForward (ff.net -> ff.lin1/lin2)
        new_key = new_key.replace('.ff.net.0.', '.ff.lin1.')
        new_key = new_key.replace('.ff.net.3.', '.ff.lin2.')
        
        # TimestepEmbedding (time_embed.time_mlp -> time_embed.time_mlp_0/1/2)
        new_key = new_key.replace('time_embed.time_mlp.0.', 'time_embed.time_mlp_0.')
        new_key = new_key.replace('time_embed.time_mlp.2.', 'time_embed.time_mlp_2.')
        
        # InputEmbedding (conv_pos_embed.conv1d -> conv_pos_embed.conv1d_0/1/2)
        new_key = new_key.replace('.conv_pos_embed.conv1d.0.', '.conv_pos_embed.conv1d_0.')
        new_key = new_key.replace('.conv_pos_embed.conv1d.2.', '.conv_pos_embed.conv1d_2.')
        
        # --- Handle wrapper class paths ---
        # e.g., "transformer_blocks.0.ff" is now "transformer_blocks.0.block.ff"
        if '.block.' not in new_key and 'transformer_blocks.' in new_key:
             new_key = new_key.replace('.attn_norm.', '.block.attn_norm.')
             new_key = new_key.replace('.attn.', '.block.attn.')
             new_key = new_key.replace('.ff_norm.', '.block.ff_norm.')
             new_key = new_key.replace('.ff.', '.block.ff.')

        new_state_dict[new_key] = value
        
    print("Key conversion complete.")
    return new_state_dict


@torch.no_grad()
def infer_process_v2(
    ref_audio_path: str, 
    ref_text: str, 
    gen_text: str, 
    model: CFMHooked, 
    vocoder, 
    device: str, 
    **kwargs
) -> tuple:
    """
    A simplified, local inference process that calls the hooked cfm.sample
    and returns the audio and the activation cache.
    """
    
    # 1. Preprocess
    target_sr = model.mel_spec.sr
    ref_audio_wav, sr = load_wav(ref_audio_path, target_sr)
    if sr != target_sr:
        print(f"Warning: Resampling ref audio from {sr} to {target_sr}")
        # This requires torchaudio, which should be installed
        from torchaudio.transforms import Resample
        resampler = Resample(sr, target_sr)
        ref_audio_wav = resampler(torch.from_numpy(ref_audio_wav)).numpy()

    ref_audio = torch.from_numpy(ref_audio_wav).unsqueeze(0).to(device)
    
    # Tokenize texts
    ref_text_tokens = list_str_to_idx([ref_text], model.vocab_char_map).to(device)
    gen_text_tokens = list_str_to_idx([gen_text], model.vocab_char_map).to(device)
    
    # Create reference mel
    mel_spec_fn = model.mel_spec.to(device)
    ref_mel = mel_spec_fn(ref_audio).permute(0, 2, 1)
    
    # 2. Run Sample
    # This is where the hooks are passed via **kwargs
    out, _, cache = model.sample(
        cond=ref_mel,
        text=gen_text_tokens,
        duration=None, # Let model predict
        **kwargs
    )
    
    # 3. Vocode
    wav = vocoder(out.permute(0, 2, 1)).squeeze().cpu().numpy()
    
    return wav, vocoder.sample_rate, out, cache


def run_ablation_experiment(
    model, 
    vocoder, 
    ref_audio_path, 
    ref_text, 
    gen_text, 
    experiment_name="baseline",
    ablation_config=None,
    **inference_kwargs
):
    """
    Run a single ablation experiment.
    REFACTORED for the new HookedRootModule system.
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*70}")
    
    # --- NEW HOOK LOGIC ---
    hook_names = []
    hook_fns = []
    capture_cache = False
    
    if ablation_config:
        print(f"Parsing {len(ablation_config)} hook config(s):")
        for hook_name, config in ablation_config.items():
            if config['type'] == 'patch':
                path = config['path']
                patch_kwargs = config.get('patch_kwargs', {})
                
                print(f"  - [PATCH] '{hook_name}' at '{path}'")
                print(f"    kwargs: {patch_kwargs}")
                
                hook_names.append(path)
                hook_fns.append(make_patch_hook(**patch_kwargs))
                
            elif config['type'] == 'extract':
                print(f"  - [EXTRACT] All activations (capture_cache=True)")
                capture_cache = True
    
    # Add hooks to inference_kwargs, which cfm_hooked.sample() accepts
    inference_kwargs['hook_names'] = hook_names
    inference_kwargs['hook_fns'] = hook_fns
    inference_kwargs['capture_cache'] = capture_cache
    # --- END NEW HOOK LOGIC ---
    
    try:
        # We use our new infer_process_v2
        audio_segment, sample_rate, spectrogram, cache = infer_process_v2(
            ref_audio_path, ref_text, gen_text, model, vocoder, **inference_kwargs
        )
        
        activation_stats = {}
        if capture_cache and cache:
            print("Collecting extracted activations from cache...")
            for name in cache.keys():
                try:
                    stats = cache.get_statistics(name)
                    activation_stats[name] = {
                        'shape': stats['shape'],
                        'mean': stats['mean'],
                        'std': stats['std'],
                        'min': stats['min'],
                        'max': stats['max'],
                    }
                except Exception as e:
                    print(f"Could not get stats for {name}: {e}")
            print(f"Captured stats for {len(activation_stats)} activations.")

        print(f"\nResult: SUCCESS")
        print(f"  Audio length: {len(audio_segment)/sample_rate:.2f}s")
        
        return {
            'audio': audio_segment,
            'sample_rate': sample_rate,
            'spectrogram': spectrogram,
            'activation_stats': activation_stats,
            'cache': cache, # Store the whole cache
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
            'cache': None,
            'success': False,
            'error': str(e)
        }


def define_ablation_experiments(num_blocks=24):
    """
    Define ablation study configurations for the NEW HookedDiT.
    Paths are updated to semantic hook names.
    """
    print(f"Defining experiments for a model with {num_blocks} blocks.")
    experiments = {
        # 1. Baseline - no modifications
        "01_baseline": None,
        
        # 2. Zero out entire transformer blocks (using residual_post hook)
        "02_zero_block_0_post": {
            "zero_b0": {
                "type": "patch",
                "path": "transformer_blocks.0.hook_resid_post",
                "patch_kwargs": {"patch_value": 0.0}
            }
        },
        "03_zero_block_last_post": {
            "zero_blast": {
                "type": "patch",
                "path": f"transformer_blocks.{num_blocks-1}.hook_resid_post",
                "patch_kwargs": {"patch_value": 0.0}
            }
        },
        
        # 3. Zero out input embeddings
        "04_zero_input_embed_proj": {
            "zero_input": {
                "type": "patch",
                "path": "input_embed.hook_proj_out",
                "patch_kwargs": {"patch_value": 0.0}
            }
        },
        
        # 4. Zero out text embeddings
        "05_zero_text_embed": {
            "zero_text": {
                "type": "patch",
                "path": "hook_text_embed",
                "patch_kwargs": {"patch_value": 0.0}
            }
        },
        
        # 5. Zero out time embedding
        "06_zero_time_embed": {
            "zero_time": {
                "type": "patch",
                "path": "time_embed.hook_mlp_out",
                "patch_kwargs": {"patch_value": 0.0}
            }
        },
        
        # 7. Constant imputation on final projection input
        "07_constant_proj_in": {
            "const_proj": {
                "type": "patch",
                "path": "hook_proj_in",
                "patch_kwargs": {"patch_value": 0.01}
            }
        },
        
        # 8. Feature extraction for analysis
        "08_extract_all_activations": {
            "extractor": {
                "type": "extract"
                # This will set capture_cache=True and get ALL hooks
            }
        },

        # --- NEW EXPERIMENTS (Only possible with new code) ---

        # 9. Ablate a single attention head (e.g., block 10, head 5)
        "09_ablate_b10_head5": {
            "ablate_h5": {
                "type": "patch",
                "path": "transformer_blocks.10.block.attn.hook_z",
                "patch_kwargs": {
                    "patch_value": 0.0,
                    "head_idx": 5
                }
            }
        },
        
        # 10. Ablate *all* Q vectors in a block (e.g., block 10)
        "10_ablate_b10_all_Q": {
            "ablate_q": {
                "type": "patch",
                "path": "transformer_blocks.10.block.attn.hook_q",
                "patch_kwargs": {"patch_value": 0.0}
            }
        },
        
        # 11. Ablate the FF block *after* the activation (e.g., block 10)
        "11_ablate_b10_ff_post_act": {
            "ablate_ff": {
                "type": "patch",
                "path": "transformer_blocks.10.block.ff.hook_post_act",
                "patch_kwargs": {"patch_value": 0.0}
            }
        },
        
        # 12. Ablate *only* the scale component of the final layernorm
        "12_ablate_norm_out_scale": {
            "ablate_scale": {
                "type": "patch",
                "path": "norm_out.hook_scale",
                "patch_kwargs": {"patch_value": 1.0} # Set scale to 1.0 (no-op)
            }
        }
    }
    
    # Filter out experiments that are out of bounds (e.g., block 23 on a 12-block model)
    valid_experiments = {}
    for name, config in experiments.items():
        is_valid = True
        if config:
            for hook_conf in config.values():
                path = hook_conf.get('path', '')
                if 'transformer_blocks.' in path:
                    try:
                        block_num = int(path.split('transformer_blocks.')[1].split('.')[0])
                        if block_num >= num_blocks:
                            is_valid = False
                            print(f"Skipping experiment '{name}': block index {block_num} out of range for model with {num_blocks} blocks.")
                            break
                    except Exception as e:
                        print(f"Warning: Could not parse block number for '{name}': {e}")
        if is_valid:
            valid_experiments[name] = config

    return valid_experiments


def save_results(results, output_dir, gen_text):
    """Save experiment results and audio files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(output_dir) / f"dit_ablation_v2_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'gen_text': gen_text,
        'experiments': {}
    }
    
    # Save audio files and collect metadata
    for exp_name, result in results.items():
        # Don't save the full cache, it's too big
        exp_metadata = {
            'success': result['success'],
            'error': result['error'],
            'activation_stats': result['activation_stats'],
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
    
    print(f"\nResults saved to: {results_dir}")
    return results_dir


def main():
    """Main ablation study runner."""
    print("\n" + "="*70)
    print("F5-TTS DiT ABLATION STUDY (v2 - Hooked)")
    print("="*70 + "\n")
    
    # Configuration
    model_name = "F5TTS_v1_Base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ---!!! PATHS FROM YOUR SCRIPT !!!---
    LOCAL_MODEL_PATH = "/home/ewong/links/scratch/checkpoints/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    LOCAL_VOCODER_PATH = "/home/ewong/links/scratch/checkpoints/vocos-mel-24khz"
    REF_AUDIO_PATH = "Original Speech and Text/84_121123_000008_000000.wav"
    REF_TEXT_PATH = "Original Speech and Text/84_121123_000008_000000.normalized.txt"
    GEN_TEXT = "This analysis captures specific layer activations across time."
    OUTPUT_DIR = "ablation_results_v2"
    # ------------------------------------
    
    # --- Validate Paths ---
    if not Path(LOCAL_MODEL_PATH).exists():
        print(f"ERROR: Model checkpoint not found at {LOCAL_MODEL_PATH}")
        return
    if not Path(LOCAL_VOCODER_PATH).exists():
        print(f"ERROR: Vocoder not found at {LOCAL_VOCODER_PATH}")
        return
    if not Path(REF_AUDIO_PATH).exists():
        print(f"ERROR: Reference audio not found at {REF_AUDIO_PATH}")
        return
    if not Path(REF_TEXT_PATH).exists():
        print(f"ERROR: Reference text not found at {REF_TEXT_PATH}")
        return
    
    with open(REF_TEXT_PATH, 'r', encoding='utf-8') as f:
        REF_TEXT_CONTENT = f.read().strip()
    
    print(f"Device: {device}")
    
    # --- NEW MODEL LOADING LOGIC ---
    print("Loading vocoder...")
    vocoder = load_vocoder(
        vocoder_name="vocos", is_local=True,
        local_path=LOCAL_VOCODER_PATH, device=device
    )
    
    print("Loading config...")
    model_cfg = OmegaConf.load(
        str(files("f5_tts").joinpath(f"configs/{model_name}.yaml"))
    )
    
    print("Instantiating Hooked Models...")
    # 1. Instantiate our Hooked DiT backbone
    # We must use the **exact** same config as the saved model
    dit_backbone = HookedDiT(**model_cfg.model.arch.transformer)
    
    # 2. Instantiate our Hooked CFM model
    model = CFMHooked(
        transformer=dit_backbone,
        mel_spec_kwargs=model_cfg.model.arch.mel_spec_kwargs,
        vocab_char_map=model_cfg.model.arch.vocab_char_map, # Pass this in
        **model_cfg.model.arch.cfm
    ).to(device)
    
    # 3. Load the *original* state dict
    print(f"Loading state dict from: {LOCAL_MODEL_PATH}")
    if LOCAL_MODEL_PATH.endswith('.safetensors'):
        from safetensors.torch import load_file
        original_state_dict = load_file(LOCAL_MODEL_PATH, device="cpu")
    else:
        original_state_dict = torch.load(LOCAL_MODEL_PATH, map_location='cpu')

    # Handle nested/EMA state dicts
    if 'model' in original_state_dict:
        original_state_dict = original_state_dict['model']
    if 'ema_model' in original_state_dict:
        original_state_dict = original_state_dict['ema_model']
    
    # 4. Convert the keys
    new_state_dict = convert_state_dict_keys(original_state_dict)
    
    # 5. Load the *new* state dict
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    print("\nState Dict Loading Report:")
    if missing_keys:
        print(f"  Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
    if not missing_keys and not unexpected_keys:
        print("  SUCCESS: State dict loaded perfectly.")
    
    model.eval()
    print("Hooked model loaded and weights converted.\n")
    # --- END NEW LOADING LOGIC ---
    
    print("Available Hook Points (Sample from Block 0):")
    dit_backbone.print_hook_points(filter_fn=lambda name: 'transformer_blocks.0' in name)
    
    print(f"Reference audio: {REF_AUDIO_PATH}")
    print(f"Reference text: {REF_TEXT_CONTENT[:50]}...")
    print(f"Generation text: {GEN_TEXT}\n")
    
    # Define experiments
    num_blocks = model.transformer.depth
    experiments = define_ablation_experiments(num_blocks=num_blocks)
    
    # Inference parameters
    inference_params = {
        'device': device,
        'nfe_step': 32,
        'cfg_strength': 2.0,
    }
    
    # --- STAGE 1 - CAPTURE BASELINE ACTIVATIONS ---
    print("\n" + "="*70)
    print("STAGE 1: Capturing baseline activations")
    print("="*70)
    
    capture_config = experiments["08_extract_all_activations"]
    capture_result = run_ablation_experiment(
        model=model, vocoder=vocoder, 
        ref_audio_path=REF_AUDIO_PATH, ref_text=REF_TEXT_CONTENT, 
        gen_text=GEN_TEXT, experiment_name="08_extract_all_activations", 
        ablation_config=capture_config, **inference_params
    )
    
    captured_cache = capture_result.get('cache')

    if captured_cache is None:
        print("ERROR: Failed to capture the activation cache. Aborting re-insertion tests.")
    else:
        print(f"Successfully captured {len(captured_cache)} activations.")
        
        # --- STAGE 2 - ADD 'RE-INSERT' EXPERIMENTS ---
        tensor_b0 = captured_cache.get_activation("transformer_blocks.0.hook_resid_post")
        
        if tensor_b0 is not None:
            print("Adding dynamic re-insertion experiments...")
            experiments["13_reinsert_block0_identity"] = {
                "reinsert_b0": {
                    "type": "patch",
                    "path": "transformer_blocks.0.hook_resid_post",
                    "patch_kwargs": {"patch_value": tensor_b0.clone()} # Clone to be safe
                }
            }
            experiments["14_reinsert_block0_to_block4"] = {
                "reinsert_b0_to_b4": {
                    "type": "patch",
                    "path": "transformer_blocks.4.hook_resid_post",
                    "patch_kwargs": {"patch_value": tensor_b0.clone()}
                }
            }
        else:
            print("Warning: Could not find 'transformer_blocks.0.hook_resid_post' in cache.")

    # --- STAGE 3 - RUN ALL EXPERIMENTS ---
    print("\n" + "="*70)
    print("STAGE 3: Running all ablation experiments")
    print("="*70)
    
    results = {
        "08_extract_all_activations": capture_result # Save the capture run
    }

    for exp_name, ablation_config in experiments.items():
        if exp_name in results: # Skip the one we already ran
            continue
            
        result = run_ablation_experiment(
            model=model,
            vocoder=vocoder,
            ref_audio_path=REF_AUDIO_PATH,
            ref_text=REF_TEXT_CONTENT,
            gen_text=GEN_TEXT,
            experiment_name=exp_name,
            ablation_config=ablation_config,
            **inference_params
        )
        results[exp_name] = result
        
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Save results
    results_dir = save_results(results, OUTPUT_DIR, GEN_TEXT)
    
    print("\n" + "="*70)
    print("ABLATION STUDY COMPLETE")
    print("="*70)
    print(f"Results saved to: {results_dir}")

if __name__ == "__main__":
    main()