# CODE FROM CLAUDE
"""
Simple attention extraction script for F5-TTS DiT model (MODIFIED FOR NEW CFM/UTILS).
Runs 3 experiments:
1. Baseline (normal inference)
2. Extract attention outputs before projection layer WITH PER-HEAD BREAKDOWN
3. Extract all layer activations (attention + feedforward)
"""

import sys
import torch
import json
from pathlib import Path
from datetime import datetime
import soundfile as sf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from f5_tts.infer.utils_infer import (
    load_model, load_vocoder, preprocess_ref_audio_text, infer_process
)
from omegaconf import OmegaConf
from hydra.utils import get_class
from importlib.resources import files


def extract_attention(model, vocoder, ref_audio, ref_text, gen_text, device):
    """Extract attention activations from DiT model using the new hooking mechanism."""
    
    # Access DiT through the model hierarchy
    cfm_model = model.ema_model if hasattr(model, 'ema_model') else model
    dit_model = cfm_model.transformer if hasattr(cfm_model, 'transformer') else cfm_model
    
    results = {}
    inference_kwargs = {
        'mel_spec_type': 'vocos', 'nfe_step': 32, 'cfg_strength': 2.0, 'device': device
    }
    
    # --- Experiment 1: Baseline ---
    print("\n" + "="*70 + "\nEXPERIMENT 1: Baseline (no hooks)\n" + "="*70)
    
    # The new infer_process returns 4 values; we ignore the last one here.
    audio, sample_rate, _, _ = infer_process(
        ref_audio, ref_text, gen_text, model, vocoder, **inference_kwargs
    )
    
    results['baseline'] = {
        'audio': audio,
        'sample_rate': sample_rate,
        'audio_length': len(audio) / sample_rate
    }
    print(f"Generated baseline audio: {len(audio)/sample_rate:.2f}s")
    
    
    # --- Experiment 2: Extract attention WITH PER-HEAD BREAKDOWN ---
    print("\n" + "="*70 + "\nEXPERIMENT 2: Extract Attention with Per-Head Statistics\n" + "="*70)
    
    # Define the specific hooks we want the model to capture for this run
    hooks_for_attention = {
        'attn_block0': {'path': 'transformer_blocks.0.attn'},
        'attn_block11': {'path': 'transformer_blocks.11.attn'},
        'attn_block21': {'path': 'transformer_blocks.21.attn'}
    }
    
    audio2, sample_rate2, _, all_timesteps2 = infer_process(
        ref_audio, ref_text, gen_text, model, vocoder,
        capture_hooks_config=hooks_for_attention, **inference_kwargs
    )
    
    # Process the activations from the FINAL timestep
    attention_stats = {}
    if all_timesteps2:
        last_timestep_key = list(all_timesteps2.keys())[-1]
        print(f"Analyzing activations from the final timestep: {last_timestep_key}")
        final_step_activations = all_timesteps2[last_timestep_key]

        for name, tensor in final_step_activations.items():
            # Get attention module to determine head configuration
            block_idx = int(re.search(r'\d+', name).group())
            attn_module = dit_model.transformer_blocks[block_idx].attn
            
            num_heads = attn_module.heads
            hidden_dim = tensor.shape[-1]
            head_dim = hidden_dim // num_heads
            batch, seq_len, _ = tensor.shape
            
            # Reshape to separate heads: [batch, seq_len, num_heads, head_dim]
            heads_tensor = tensor.reshape(batch, seq_len, num_heads, head_dim)
            
            attention_stats[name] = {
                'shape': list(tensor.shape),
                'mean': float(tensor.mean()), 'std': float(tensor.std()),
                'min': float(tensor.min()), 'max': float(tensor.max()),
                'num_heads': num_heads, 'head_dim': head_dim,
                'per_head_stats': {}
            }
            
            for head_idx in range(num_heads):
                head_tensor = heads_tensor[:, :, head_idx, :]
                attention_stats[name]['per_head_stats'][f'head_{head_idx}'] = {
                    'mean': float(head_tensor.mean()), 'std': float(head_tensor.std()),
                    'min': float(head_tensor.min()), 'max': float(head_tensor.max()),
                    'norm': float(torch.norm(head_tensor))
                }
            
            print(f"\n{name}:")
            print(f"  Overall - Mean: {tensor.mean():.6f}, Std: {tensor.std():.6f}")
            for head_idx in range(min(3, num_heads)): # Print first 3 heads
                stats = attention_stats[name]['per_head_stats'][f'head_{head_idx}']
                print(f"    Head {head_idx}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, norm={stats['norm']:.4f}")
    
    results['attention_extraction'] = {
        'audio': audio2, 'sample_rate': sample_rate2, 'statistics': attention_stats
    }
    
    
    # --- Experiment 3: Extract all layers ---
    print("\n" + "="*70 + "\nEXPERIMENT 3: Extract All Layers (attention + feedforward)\n" + "="*70)
    
    hooks_for_all_layers = {
        'input_embed': {'path': 'input_embed'},
        'block0': {'path': 'transformer_blocks.0'},
        'block0_attn': {'path': 'transformer_blocks.0.attn'},
        'block0_ff': {'path': 'transformer_blocks.0.ff'},
        'block11': {'path': 'transformer_blocks.11'},
        'block21': {'path': 'transformer_blocks.21'}
    }
    
    audio3, sample_rate3, _, all_timesteps3 = infer_process(
        ref_audio, ref_text, gen_text, model, vocoder,
        capture_hooks_config=hooks_for_all_layers, **inference_kwargs
    )
    
    all_stats = {}
    if all_timesteps3:
        last_timestep_key = list(all_timesteps3.keys())[-1]
        print(f"Analyzing activations from the final timestep: {last_timestep_key}")
        final_step_activations = all_timesteps3[last_timestep_key]

        for name, tensor in final_step_activations.items():
            all_stats[name] = {
                'shape': list(tensor.shape),
                'mean': float(tensor.mean()), 'std': float(tensor.std()),
                'min': float(tensor.min()), 'max': float(tensor.max())
            }
            print(f"\n{name}: Shape: {list(tensor.shape)}, Mean: {tensor.mean():.6f}, Std: {tensor.std():.6f}")

    results['all_layers'] = {
        'audio': audio3, 'sample_rate': sample_rate3, 'statistics': all_stats
    }
    
    return results


def save_results(results, output_dir="attention_check_results"):
    """Save audio files and statistics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(output_dir) / f"attention_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for exp_name, data in results.items():
        if 'audio' in data and data['audio'] is not None:
            sf.write(results_dir / f"{exp_name}.wav", data['audio'], data['sample_rate'])
    
    stats_to_save = {exp: data.get('statistics', {}) for exp in results}
    stats_path = results_dir / "attention_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats_to_save, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")
    return results_dir


def main():
    print("\n" + "="*70 + "\nF5-TTS ATTENTION EXTRACTION CHECK\n" + "="*70)
    
    import re # Import re for Experiment 2
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "F5TTS_v1_Base"
    
    LOCAL_MODEL_PATH = "/home/ewong/links/scratch/F5-TTS/checkpoints/F5-TTS/F5TTS_v1_Base/model_1250000.safensors"
    LOCAL_VOCODER_PATH = "/home/ewong/links/scratch/F5-TTS/checkpoints/vocos-mel-24khz"
    
    print(f"\nDevice: {device}")
    vocoder = load_vocoder(is_local=True, local_path=LOCAL_VOCODER_PATH, device=device)
    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model_name}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model = load_model(model_cls, model_cfg.model.arch, LOCAL_MODEL_PATH, mel_spec_type="vocos", device=device)
    
    ref_audio_path = "84_121123_000008_000000.wav"
    ref_text_path = "84_121123_000008_000000.normalized.txt"
    with open(ref_text_path, 'r') as f:
        ref_text_content = f.read().strip()
    
    gen_text = "This is a test of the attention mechanism."
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text_content)
    
    results = extract_attention(model, vocoder, ref_audio, ref_text, gen_text, device)
    save_results(results)
    
    print("\n" + "="*70 + "\nCOMPLETE\n" + "="*70)

if __name__ == "__main__":
    main()