# Code FROM CLAUDE
"""
Simple attention extraction script for F5-TTS DiT model.
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
    """Extract attention activations from DiT model."""
    
    # Access DiT through the model hierarchy
    cfm_model = model.ema_model if hasattr(model, 'ema_model') else model
    dit_model = cfm_model.transformer if hasattr(cfm_model, 'transformer') else cfm_model
    
    results = {}
    
    # Experiment 1: Baseline
    print("\n" + "="*70)
    print("EXPERIMENT 1: Baseline (no hooks)")
    print("="*70)
    
    audio, sample_rate, spectrogram = infer_process(
        ref_audio, ref_text, gen_text, model, vocoder,
        mel_spec_type='vocos', target_rms=0.1, cross_fade_duration=0.15,
        nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1,
        speed=1.0, fix_duration=None, device=device
    )
    
    results['baseline'] = {
        'audio': audio,
        'sample_rate': sample_rate,
        'audio_length': len(audio) / sample_rate
    }
    print(f"Generated audio: {len(audio)/sample_rate:.2f}s")
    
    
    # Experiment 2: Extract attention WITH PER-HEAD BREAKDOWN
    print("\n" + "="*70)
    print("EXPERIMENT 2: Extract Attention with Per-Head Statistics")
    print("="*70)
    
    dit_model.clear_activations()
    dit_model.remove_hooks()
    
    # Add hooks to attention layers
    dit_model.add_extraction_hook('transformer_blocks.0.attn', 'attn_block0')
    dit_model.add_extraction_hook('transformer_blocks.11.attn', 'attn_block11')
    dit_model.add_extraction_hook('transformer_blocks.23.attn', 'attn_block23')
    
    audio2, sample_rate2, spectrogram2 = infer_process(
        ref_audio, ref_text, gen_text, model, vocoder,
        mel_spec_type='vocos', target_rms=0.1, cross_fade_duration=0.15,
        nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1,
        speed=1.0, fix_duration=None, device=device
    )
    
    # Collect attention statistics WITH PER-HEAD BREAKDOWN
    attention_stats = {}
    for name, data in dit_model.extracted_activations.items():
        tensor = data['tensor']  # Shape: [batch, seq_len, hidden_dim]
        
        # Get attention module to determine head configuration
        block_idx = int(name.split('_block')[1])
        attn_module = dit_model.transformer_blocks[block_idx].attn
        
        # Attention config
        num_heads = attn_module.heads if hasattr(attn_module, 'heads') else 8
        hidden_dim = tensor.shape[-1]
        head_dim = hidden_dim // num_heads
        
        batch, seq_len, _ = tensor.shape
        
        # Reshape to separate heads: [batch, seq_len, num_heads, head_dim]
        heads_tensor = tensor.reshape(batch, seq_len, num_heads, head_dim)
        
        # Statistics for whole attention output
        attention_stats[name] = {
            'shape': list(data['shape']),
            'mean': float(tensor.mean()),
            'std': float(tensor.std()),
            'min': float(tensor.min()),
            'max': float(tensor.max()),
            'num_heads': num_heads,
            'head_dim': head_dim,
            'per_head_stats': {}
        }
        
        # Per-head statistics
        for head_idx in range(num_heads):
            head_tensor = heads_tensor[:, :, head_idx, :]  # [batch, seq_len, head_dim]
            attention_stats[name]['per_head_stats'][f'head_{head_idx}'] = {
                'mean': float(head_tensor.mean()),
                'std': float(head_tensor.std()),
                'min': float(head_tensor.min()),
                'max': float(head_tensor.max()),
                'norm': float(torch.norm(head_tensor))
            }
        
        print(f"\n{name}:")
        print(f"  Shape: {data['shape']}")
        print(f"  Num heads: {num_heads}, Head dim: {head_dim}")
        print(f"  Overall - Mean: {tensor.mean():.6f}, Std: {tensor.std():.6f}")
        print(f"  Per-head statistics:")
        for head_idx in range(min(3, num_heads)):  # Print first 3 heads
            stats = attention_stats[name]['per_head_stats'][f'head_{head_idx}']
            print(f"    Head {head_idx}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, norm={stats['norm']:.4f}")
        if num_heads > 3:
            print(f"    ... ({num_heads - 3} more heads)")
    
    results['attention_extraction'] = {
        'audio': audio2,
        'sample_rate': sample_rate2,
        'statistics': attention_stats
    }
    
    dit_model.remove_hooks()
    
    
    # Experiment 3: Extract all layers
    print("\n" + "="*70)
    print("EXPERIMENT 3: Extract All Layers (attention + feedforward)")
    print("="*70)
    
    dit_model.clear_activations()
    
    # Add comprehensive hooks
    dit_model.add_extraction_hook('input_embed', 'input_embed')
    dit_model.add_extraction_hook('transformer_blocks.0', 'block0')
    dit_model.add_extraction_hook('transformer_blocks.0.attn', 'block0_attn')
    dit_model.add_extraction_hook('transformer_blocks.0.ff', 'block0_ff')
    dit_model.add_extraction_hook('transformer_blocks.11', 'block11')
    dit_model.add_extraction_hook('transformer_blocks.11.attn', 'block11_attn')
    dit_model.add_extraction_hook('transformer_blocks.23', 'block23')
    dit_model.add_extraction_hook('transformer_blocks.23.attn', 'block23_attn')
    
    audio3, sample_rate3, spectrogram3 = infer_process(
        ref_audio, ref_text, gen_text, model, vocoder,
        mel_spec_type='vocos', target_rms=0.1, cross_fade_duration=0.15,
        nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1,
        speed=1.0, fix_duration=None, device=device
    )
    
    # Collect all statistics
    all_stats = {}
    for name, data in dit_model.extracted_activations.items():
        tensor = data['tensor']
        all_stats[name] = {
            'shape': list(data['shape']),
            'mean': float(tensor.mean()),
            'std': float(tensor.std()),
            'min': float(tensor.min()),
            'max': float(tensor.max())
        }
        print(f"\n{name}:")
        print(f"  Shape: {data['shape']}")
        print(f"  Mean: {tensor.mean():.6f}")
        print(f"  Std:  {tensor.std():.6f}")
    
    results['all_layers'] = {
        'audio': audio3,
        'sample_rate': sample_rate3,
        'statistics': all_stats
    }
    
    dit_model.remove_hooks()
    
    return results


def save_results(results, output_dir="attention_check_results"):
    """Save audio files and statistics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(output_dir) / f"attention_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save audio files
    for exp_name, data in results.items():
        if 'audio' in data:
            audio_path = results_dir / f"{exp_name}.wav"
            sf.write(audio_path, data['audio'], data['sample_rate'])
            print(f"\nSaved: {audio_path}")
    
    # Save statistics
    stats = {}
    for exp_name, data in results.items():
        if 'statistics' in data:
            stats[exp_name] = data['statistics']
        else:
            stats[exp_name] = {
                'audio_length_sec': data.get('audio_length', 0)
            }
    
    stats_path = results_dir / "attention_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved: {stats_path}")
    
    return results_dir


def main():
    print("\n" + "="*70)
    print("F5-TTS ATTENTION EXTRACTION CHECK")
    print("="*70)
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "F5TTS_v1_Base"
    
    LOCAL_MODEL_PATH = "/home/ewong/links/scratch/F5-TTS/checkpoints/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    LOCAL_VOCODER_PATH = "/home/ewong/links/scratch/F5-TTS/checkpoints/vocos-mel-24khz"
    
    # Check files
    if not Path(LOCAL_MODEL_PATH).exists():
        print(f"ERROR: Model not found at {LOCAL_MODEL_PATH}")
        return
    if not Path(LOCAL_VOCODER_PATH).exists():
        print(f"ERROR: Vocoder not found at {LOCAL_VOCODER_PATH}")
        return
    
    print(f"\nDevice: {device}")
    print("Loading vocoder...")
    vocoder = load_vocoder(
        vocoder_name="vocos",
        is_local=True,
        local_path=LOCAL_VOCODER_PATH,
        device=device
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
    
    # Setup reference data
    ref_audio = "84_121123_000008_000000.wav"
    ref_text_file = "84_121123_000008_000000.normalized.txt"
    
    if not Path(ref_audio).exists() or not Path(ref_text_file).exists():
        print(f"ERROR: Reference files not found")
        return
    
    with open(ref_text_file, 'r') as f:
        ref_text = f.read().strip()
    
    gen_text = "This is a test of the attention mechanism."
    
    print(f"\nReference: {ref_audio}")
    print(f"Text to generate: {gen_text}")
    
    # Preprocess
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text)
    
    # Run extraction
    results = extract_attention(model, vocoder, ref_audio, ref_text, gen_text, device)
    
    # Save results
    results_dir = save_results(results)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {results_dir}")
    print("\nCheck attention_statistics.json for detailed per-head statistics")
    print("\nTo view per-head data:")
    print(f"  cat {results_dir}/attention_statistics.json | jq '.attention_extraction'")


if __name__ == "__main__":
    main()