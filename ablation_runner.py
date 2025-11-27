#!/usr/bin/env python3
"""
F5-TTS DiT Ablation Study Runner
Tests DiT model with ablations, attention head patching, and causal interventions.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import soundfile as sf

from f5_tts.model.cfm import HookConfig, HookManager


def run_ablation_experiment(
    model,
    vocoder,
    ref_audio,
    ref_text,
    gen_text,
    experiment_name: str = "baseline",
    ablation_config: dict | None = None,
    **inference_kwargs,
):
    from f5_tts.infer.utils_infer import infer_process

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*70}")

    # Resolve the underlying DiT model inside the top-level model wrapper
    dit_model = None
    if hasattr(model, "ema_model"):
        cfm_model = model.ema_model
        if hasattr(cfm_model, "transformer"):
            dit_model = cfm_model.transformer
        else:
            dit_model = cfm_model
    elif hasattr(model, "transformer"):
        dit_model = model.transformer
    else:
        dit_model = model

    print(f"DiT model type: {type(dit_model).__name__}")

    # ------------------------------------------------------------------
    # Build HookConfig based on ablation type
    # ------------------------------------------------------------------
    capture_layers_dict: dict[str, str] = {}
    patches_dict: dict[int, list[dict]] = {}
    is_extracting = False

    if ablation_config:
        ablation_type = ablation_config["type"]
        path = ablation_config["path"]

        print(f"Ablation type: {ablation_type}")
        print(f"Target: {path}")

        if ablation_type == "extract":
            capture_layers_dict[experiment_name] = path
            is_extracting = True
            print(f"  → Extracting activations from {path}")

        elif ablation_type == "zero_all_steps":
            patch_kwargs = {
                "patch_vec": 0.0,
                "batch_index": ablation_config.get("batch_index"),
                "feature_index": ablation_config.get("feature_index"),
                "extract_name_before": f"{experiment_name}.before.{path}",
                "extract_name_after": f"{experiment_name}.after.{path}",
            }

            nfe_steps = inference_kwargs.get("nfe_step", 32)
            for step in range(nfe_steps + 1):
                patches_dict.setdefault(step, []).append(
                    {"path": path, "kwargs": patch_kwargs}
                )
            print(f"  → Zeroing at ALL {nfe_steps + 1} steps")

        elif ablation_type == "constant_all_steps":
            patch_kwargs = {
                "patch_vec": ablation_config["value"],
                "batch_index": ablation_config.get("batch_index"),
                "feature_index": ablation_config.get("feature_index"),
                "extract_name_before": f"{experiment_name}.before.{path}",
                "extract_name_after": f"{experiment_name}.after.{path}",
            }

            nfe_steps = inference_kwargs.get("nfe_step", 32)
            for step in range(nfe_steps + 1):
                patches_dict.setdefault(step, []).append(
                    {"path": path, "kwargs": patch_kwargs}
                )
            print(f"  → Constant {ablation_config['value']} at ALL {nfe_steps + 1} steps")

        elif ablation_type == "patch_at_step":
            patch_steps = ablation_config.get("patch_steps", [ablation_config.get("patch_step", 10)])
            if not isinstance(patch_steps, list):
                patch_steps = [patch_steps]

            patch_kwargs = {
                "patch_vec": ablation_config.get("value", 0.0),
                "batch_index": ablation_config.get("batch_index"),
                "feature_index": ablation_config.get("feature_index"),
                "extract_name_before": f"{experiment_name}.before.{path}",
                "extract_name_after": f"{experiment_name}.after.{path}",
            }

            for step in patch_steps:
                patches_dict.setdefault(step, []).append(
                    {"path": path, "kwargs": patch_kwargs}
                )
            print(f"  → Patching at steps: {patch_steps}")

        elif ablation_type == "zero_attention_head":
            # PERFORMANCE FIX: Use list indices instead of looping
            head_idx = ablation_config.get("head_index", 0)
            component = ablation_config.get("attention_component", "q")
            head_dim = ablation_config.get("head_dim", 64)

            component_map = {
                "q": f"{path}.attn.to_q",
                "k": f"{path}.attn.to_k",
                "v": f"{path}.attn.to_v",
                "o": f"{path}.attn.to_out.0",
            }
            target_path = component_map.get(component, f"{path}.attn.to_q")

            start_idx = head_idx * head_dim
            end_idx = start_idx + head_dim
            feature_indices = list(range(start_idx, end_idx))

            nfe_steps = inference_kwargs.get("nfe_step", 32)
            for step in range(nfe_steps + 1):
                patch_kwargs = {
                    "patch_vec": 0.0,
                    "feature_index": feature_indices,  # Pass list once
                    "extract_name_before": f"{experiment_name}.before.{target_path}.head{head_idx}",
                    "extract_name_after": f"{experiment_name}.after.{target_path}.head{head_idx}",
                }
                patches_dict.setdefault(step, []).append(
                    {"path": target_path, "kwargs": patch_kwargs}
                )

            print(f"  → Zeroing attention head {head_idx} ({component}) in {path}")

        elif ablation_type == "re_insert":
            source_tensor = ablation_config.get("source_tensor")
            if source_tensor is None:
                print("  ERROR: No source_tensor provided for re-insertion")
                return None

            patch_steps = ablation_config.get("patch_steps", list(range(33)))
            for step in patch_steps:
                patch_kwargs = {
                    "patch_vec": source_tensor,
                    "batch_index": ablation_config.get("batch_index"),
                    "feature_index": ablation_config.get("feature_index"),
                    "extract_name_before": f"{experiment_name}.before.{path}",
                    "extract_name_after": f"{experiment_name}.after.{path}",
                }
                patches_dict.setdefault(step, []).append(
                    {"path": path, "kwargs": patch_kwargs}
                )
            print(f"  → Re-inserting tensor into {path} at {len(patch_steps)} steps")

        elif ablation_type == "patch_attention_pattern":
            layer_idx = ablation_config.get("layer_index", 0)
            target_path = f"transformer_blocks.{layer_idx}.attn"

            nfe_steps = inference_kwargs.get("nfe_step", 32)
            for step in range(nfe_steps + 1):
                patch_kwargs = {
                    "patch_vec": 0.0,
                    "extract_name_before": f"{experiment_name}.before.{target_path}",
                    "extract_name_after": f"{experiment_name}.after.{target_path}",
                }
                patches_dict.setdefault(step, []).append(
                    {"path": target_path, "kwargs": patch_kwargs}
                )
            print(f"  → Patching attention pattern in layer {layer_idx}")

    # ------------------------------------------------------------------
    # Create HookConfig + HookManager
    # ------------------------------------------------------------------
    hook_cfg = HookConfig(
        capture_layers=capture_layers_dict,
        patches=patches_dict,
        save_per_timestep=False,
        clear_activations_per_step=False,  # Keep activations for stats!
    )

    hook_manager = HookManager(hook_cfg)
    inference_kwargs["hook_manager"] = hook_manager

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    try:
        audio_segment, sample_rate, spectrogram = infer_process(
            ref_audio,
            ref_text,
            gen_text,
            model,
            vocoder,
            **inference_kwargs,
        )

        # ------------------------------------------------------------------
        # Collect activation statistics
        # ------------------------------------------------------------------
        activation_stats: dict = {}
        captured_tensors: dict = {}

        extracted = getattr(dit_model, "extracted_activations", {})

        if not extracted and is_extracting:
            print("Warning: extracted_activations was empty or missing.")

        for name, data in extracted.items():
            if "tensor" not in data:
                continue

            tensor_copy = data["tensor"].detach().clone()

            if is_extracting:
                captured_tensors[name] = {
                    "tensor": tensor_copy,
                    "shape": data.get("shape", tuple(tensor_copy.shape)),
                    "device": data.get("device"),
                    "call_count": data.get("call_count"),
                }

            activation_stats[name] = {
                "shape": list(tensor_copy.shape),
                "mean": float(tensor_copy.mean()),
                "std": float(tensor_copy.std()),
                "min": float(tensor_copy.min()),
                "max": float(tensor_copy.max()),
                "phase": (
                    "before" if ".before." in name
                    else "after" if ".after." in name
                    else "raw"
                ),
            }

        if hasattr(dit_model, "clear_activations"):
            dit_model.clear_activations()

        # REMOVED GC HERE
        
        print(f"\nResult: SUCCESS")
        print(f"  Audio length: {len(audio_segment) / sample_rate:.2f}s")
        if activation_stats:
            print(f"  Captured {len(activation_stats)} activations")

        return {
            "audio": audio_segment,
            "sample_rate": sample_rate,
            "spectrogram": spectrogram,
            "activation_stats": activation_stats,
            "captured_tensors": captured_tensors,
            "success": True,
            "error": None,
        }

    except Exception as e:
        print(f"\nResult: FAILED - {e}")
        import traceback
        traceback.print_exc()

        if hasattr(dit_model, "clear_activations"):
            dit_model.clear_activations()
        
        # REMOVED GC HERE

        return {
            "audio": None,
            "sample_rate": None,
            "spectrogram": None,
            "activation_stats": {},
            "captured_tensors": {},
            "success": False,
            "error": str(e),
        }

# ... [define_ablation_experiments, add_reinsertion_experiments, save_results remain the same] ...

def define_ablation_experiments() -> dict[str, dict | None]:
    """Define a suite of ablation experiments over DiT."""
    experiments: dict[str, dict | None] = {
        "01_baseline": None,
        "02_zero_block0": {"type": "zero_all_steps", "path": "transformer_blocks.0"},
        "03_zero_block5": {"type": "zero_all_steps", "path": "transformer_blocks.5"},
        "04_zero_block11": {"type": "zero_all_steps", "path": "transformer_blocks.11"},
        "05_zero_block21": {"type": "zero_all_steps", "path": "transformer_blocks.21"},
        "06_zero_input_embed": {"type": "zero_all_steps", "path": "input_embed"},
        "07_zero_time_embed": {"type": "zero_all_steps", "path": "time_embed"},
        "08_zero_text_embed": {"type": "zero_all_steps", "path": "text_embed"},
        "09_zero_attn_q_block5": {"type": "zero_all_steps", "path": "transformer_blocks.5.attn.to_q"},
        "10_zero_attn_k_block5": {"type": "zero_all_steps", "path": "transformer_blocks.5.attn.to_k"},
        "11_zero_attn_v_block5": {"type": "zero_all_steps", "path": "transformer_blocks.5.attn.to_v"},
        "12_zero_head0_block5": {
            "type": "zero_attention_head",
            "path": "transformer_blocks.5",
            "head_index": 0,
            "attention_component": "q",
            "head_dim": 64,
        },
        "13_zero_head3_block5": {
            "type": "zero_attention_head",
            "path": "transformer_blocks.5",
            "head_index": 3,
            "attention_component": "v",
            "head_dim": 64,
        },
        "14_zero_feature_dim0": {"type": "zero_all_steps", "path": "transformer_blocks.5", "feature_index": 0},
        "15_zero_feature_dims_0to10": {"type": "zero_all_steps", "path": "transformer_blocks.5", "feature_index": 0},
        "16_constant_block0": {"type": "constant_all_steps", "path": "transformer_blocks.0", "value": 0.01},
        "17_constant_block5_neg": {"type": "constant_all_steps", "path": "transformer_blocks.5", "value": -0.01},
        "18_zero_block0_step10": {"type": "patch_at_step", "path": "transformer_blocks.0", "value": 0.0, "patch_step": 10},
        "19_zero_block0_early_steps": {"type": "patch_at_step", "path": "transformer_blocks.0", "value": 0.0, "patch_steps": [0, 1, 2, 3, 4]},
        "20_zero_block0_late_steps": {"type": "patch_at_step", "path": "transformer_blocks.0", "value": 0.0, "patch_steps": [28, 29, 30, 31, 32]},
        "21_zero_block0_middle_steps": {"type": "patch_at_step", "path": "transformer_blocks.0", "value": 0.0, "patch_steps": [10, 15, 20]},
        "22_extract_block0": {"type": "extract", "path": "transformer_blocks.0"},
        "23_extract_block11": {"type": "extract", "path": "transformer_blocks.11"},
        "24_extract_block21": {"type": "extract", "path": "transformer_blocks.21"},
        "25_extract_attn_q_block5": {"type": "extract", "path": "transformer_blocks.5.attn.to_q"},
    }
    return experiments

def add_reinsertion_experiments(experiments: dict, captured_tensors: dict, device: str):
    block0_tensor = None
    block0_key = None
    for key, value in captured_tensors.items():
        if "block0" in key or "block_0" in key or "transformer_blocks.0" in key:
            block0_tensor = value["tensor"].to(device)
            block0_key = key
            break

    if block0_tensor is not None:
        print(f"\nAdding re-insertion experiments using captured tensor: {block0_key}")
        experiments["26_reinsert_block0_identity"] = {
            "type": "re_insert", "path": "transformer_blocks.0", "source_tensor": block0_tensor, "patch_steps": list(range(33))
        }
        experiments["27_reinsert_block0_to_block5"] = {
            "type": "re_insert", "path": "transformer_blocks.5", "source_tensor": block0_tensor, "patch_steps": list(range(33))
        }
        experiments["28_reinsert_block0_to_block11"] = {
            "type": "re_insert", "path": "transformer_blocks.11", "source_tensor": block0_tensor, "patch_steps": list(range(33))
        }
        experiments["29_reinsert_block0_late_only"] = {
            "type": "re_insert", "path": "transformer_blocks.0", "source_tensor": block0_tensor, "patch_steps": [25, 26, 27, 28, 29, 30, 31, 32]
        }
        if block0_tensor.ndim >= 2:
            partial_tensor = block0_tensor.clone()
            partial_tensor[:, :, partial_tensor.shape[-1] // 2 :] = 0
            experiments["30_reinsert_block0_partial_features"] = {
                "type": "re_insert", "path": "transformer_blocks.0", "source_tensor": partial_tensor, "patch_steps": list(range(33))
            }
    return experiments

def save_results(results: dict, output_dir: str | Path, gen_text: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(output_dir) / f"dit_ablation_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict = {"timestamp": timestamp, "gen_text": gen_text, "experiments": {}}

    for exp_name, result in results.items():
        exp_metadata = {
            "success": result["success"],
            "error": result["error"],
            "activation_stats": result["activation_stats"],
        }
        if result["success"] and result["audio"] is not None:
            audio_path = results_dir / f"{exp_name}.wav"
            sf.write(audio_path, result["audio"], result["sample_rate"])
            exp_metadata.update({
                "audio_file": audio_path.name,
                "sample_rate": result["sample_rate"],
                "audio_length_sec": len(result["audio"]) / result["sample_rate"],
            })
        metadata["experiments"][exp_name] = exp_metadata

    with open(results_dir / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(results_dir / "summary.txt", "w") as f:
        f.write("DiT Ablation Study Results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Generated Text: {gen_text}\n\n")
        successful = sum(1 for r in results.values() if r["success"])
        f.write(f"Experiments: {successful}/{len(results)} successful\n\n")
        for exp_name, result in results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Status: {'SUCCESS' if result['success'] else 'FAILED'}\n")
            if result["error"]: f.write(f"Error: {result['error']}\n")
            if result["activation_stats"]:
                f.write("\nActivation Statistics:\n")
                for name, stats in result["activation_stats"].items():
                    f.write(f"  {name}:\n")
                    if "error" in stats:
                        f.write(f"    Error: {stats['error']}\n")
                        continue
                    f.write(f"    Mean: {stats['mean']:.6f}\n")
                    f.write(f"    Std: {stats['std']:.6f}\n")

    print(f"\nResults saved to: {results_dir}")
    return results_dir


def main():
    """Main entry point for the DiT ablation study runner."""
    print("\n" + "=" * 70)
    print("F5-TTS DiT ABLATION STUDY (ENHANCED)")
    print("=" * 70 + "\n")

    from f5_tts.infer.utils_infer import (
        load_model,
        load_vocoder,
        preprocess_ref_audio_text,
    )
    from omegaconf import OmegaConf
    from hydra.utils import get_class
    from importlib.resources import files

    model_name = "F5TTS_v1_Base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    LOCAL_MODEL_PATH = "/home/ewong/links/scratch/checkpoints/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    LOCAL_VOCODER_PATH = "/home/ewong/links/scratch/checkpoints/vocos-mel-24khz"

    if not Path(LOCAL_MODEL_PATH).exists():
        print(f"ERROR: Model not found at {LOCAL_MODEL_PATH}")
        return

    print("Loading vocoder...")
    vocoder = load_vocoder(vocoder_name="vocos", is_local=True, local_path=LOCAL_VOCODER_PATH, device=device)
    
    print("Loading F5-TTS model...")
    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model_name}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model = load_model(model_cls, model_cfg.model.arch, LOCAL_MODEL_PATH, mel_spec_type="vocos", vocab_file="", device=device)

    ref_audio = "Original Speech and Text/84_121123_000008_000000.wav"
    ref_text_file = "Original Speech and Text/84_121123_000008_000000.normalized.txt"
    if not Path(ref_audio).exists() or not Path(ref_text_file).exists():
        print("ERROR: Reference files not found")
        return
    
    with open(ref_text_file, "r", encoding="utf-8") as f:
        ref_text = f.read().strip()
    
    gen_text = "This is a test of the enhanced ablation system with attention head patching."
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text)

    inference_params = {
        "mel_spec_type": "vocos", "target_rms": 0.1, "cross_fade_duration": 0.15, "nfe_step": 32,
        "cfg_strength": 2.0, "sway_sampling_coef": -1, "speed": 1.0, "fix_duration": None, "device": device,
    }

    print("\n" + "=" * 70)
    print("STAGE 1: Capturing baseline activations")
    print("=" * 70)
    
    capture_config = {"type": "extract", "path": "transformer_blocks.0"}
    capture_result = run_ablation_experiment(
        model=model, vocoder=vocoder, ref_audio=ref_audio, ref_text=ref_text, gen_text=gen_text,
        experiment_name="00_capture_block0", ablation_config=capture_config, **inference_params,
    )
    captured_tensors = capture_result.get("captured_tensors", {})

    print("\n" + "=" * 70)
    print("STAGE 2: Running ablation experiments")
    print("=" * 70)

    experiments = define_ablation_experiments()
    if captured_tensors:
        experiments = add_reinsertion_experiments(experiments, captured_tensors, device)

    results: dict = {"00_capture_block0": capture_result}

    for exp_name, ablation_config in experiments.items():
        result = run_ablation_experiment(
            model=model, vocoder=vocoder, ref_audio=ref_audio, ref_text=ref_text, gen_text=gen_text,
            experiment_name=exp_name, ablation_config=ablation_config, **inference_params,
        )
        results[exp_name] = result

    output_dir = "ablation_results"
    results_dir = save_results(results, output_dir, gen_text)
    print("ABLATION STUDY COMPLETE")

if __name__ == "__main__":
    main()