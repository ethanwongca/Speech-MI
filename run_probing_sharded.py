#!/usr/bin/env python3
"""
Speaker Probing over DiT Hidden States and Attention Heads
==========================================================

Pipeline overview
-----------------
This script operates on WebDataset shards produced by your extraction pipeline.
It trains linear probes to classify speakers from DiT internal activations.
"""

import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import webdataset as wds
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from omegaconf import OmegaConf
from hydra.utils import get_class
from importlib.resources import files
from f5_tts.infer.utils_infer import load_model


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def load_dit_model_for_config(
    model_name: str,
    checkpoint_path: Path,
    device: str = "cpu",
) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    model_cfg = OmegaConf.load(
        str(files("f5_tts").joinpath(f"configs/{model_name}.yaml"))
    )
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arch = model_cfg.model.arch

    model = load_model(
        model_cls,
        model_arch,
        str(checkpoint_path),
        mel_spec_type="vocos",
        vocab_file="",
        device=device,
    )

    cfm_model = model.ema_model if hasattr(model, "ema_model") else model
    dit_model = cfm_model.transformer if hasattr(cfm_model, "transformer") else cfm_model

    return dit_model


def infer_attention_head_config(
    model_name: str,
    checkpoint_path: Path,
    device: str = "cpu",
) -> Tuple[int, int]:
    dit_model = load_dit_model_for_config(model_name, checkpoint_path, device=device)

    if not hasattr(dit_model, "transformer_blocks") or len(dit_model.transformer_blocks) == 0:
        raise ValueError("DiT model has no transformer blocks.")

    attn_module = dit_model.transformer_blocks[0].attn
    num_heads = int(attn_module.heads)

    if hasattr(attn_module, "dim_head"):
        head_dim = int(attn_module.dim_head)
    elif hasattr(attn_module, "dim"):
        head_dim = int(attn_module.dim) // num_heads
    else:
        raise AttributeError("Cannot infer head_dim from attention module.")

    return num_heads, head_dim


# ---------------------------------------------------------------------------
# Data loading from WebDataset
# ---------------------------------------------------------------------------

def load_tensors_from_webdataset_shards(
    shard_dir: Path,
    speakers_list: List[str],
    target_step: int,
    target_layer: str,
    max_length_sec: float = 5.0,
    frames_per_sec: int = 50,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[Tuple[int, int]]]]:
    """
    Load hidden states for a specific timestep and layer.
    """
    speaker_map = {name: i for i, name in enumerate(speakers_list)}
    max_frames = int(max_length_sec * frames_per_sec)

    X_data: List[torch.Tensor] = []
    y_data: List[torch.Tensor] = []
    speaker_boundaries: List[Tuple[int, int]] = []

    for speaker_id in speakers_list:
        if speaker_id not in speaker_map:
            continue

        speaker_label = speaker_map[speaker_id]
        pattern = f"speaker_{speaker_id}_step_{target_step:04d}_*.tar"
        matching_shards = list(shard_dir.glob(pattern))

        if not matching_shards:
            continue

        speaker_start_idx = len(X_data)

        for shard_path in matching_shards:
            try:
                dataset = wds.WebDataset(str(shard_path))

                for sample in dataset:
                    metadata = json.loads(sample["json"].decode("utf-8"))
                    layer_group = metadata.get("layer_group", "")

                    if layer_group != target_layer:
                        continue

                    tensor_data_bytes = sample["pth"]
                    data_dict = torch.load(
                        io.BytesIO(tensor_data_bytes),
                        map_location="cpu",
                    )

                    if isinstance(data_dict, dict) and "activations" in data_dict:
                        activations = data_dict["activations"]
                        # With updated CFM, each file has just "__output__"
                        if "__output__" in activations:
                            tensor = activations["__output__"]
                        else:
                            # Fallback for legacy files or edge cases
                            tensor = list(activations.values())[0]
                    else:
                        tensor = data_dict

                    if tensor.ndim == 3:
                        tensor = tensor.reshape(-1, tensor.shape[-1])
                    
                    tensor = tensor[:max_frames, :]
                    if tensor.shape[0] == 0:
                        continue

                    num_tokens = tensor.shape[0]
                    labels = torch.full(
                        (num_tokens,),
                        fill_value=speaker_label,
                        dtype=torch.long,
                    )

                    X_data.append(tensor.to(dtype=torch.float32))
                    y_data.append(labels)

            except Exception as e:
                print(f"Warning: Error loading shard {shard_path.name}: {e}")
                continue

        speaker_end_idx = len(X_data)
        if speaker_end_idx > speaker_start_idx:
            speaker_boundaries.append((speaker_start_idx, speaker_end_idx))

    if not X_data:
        return None, None, None

    X_combined = torch.cat(X_data, dim=0).numpy()
    y_combined = torch.cat(y_data, dim=0).numpy()

    return X_combined, y_combined, speaker_boundaries


# ---------------------------------------------------------------------------
# Steering vectors
# ---------------------------------------------------------------------------

def compute_steering_vectors(
    X_data: np.ndarray,
    y_data: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    feature_dim = X_data.shape[1]
    steering_vectors = np.zeros((num_classes, feature_dim), dtype=np.float32)

    for class_idx in range(num_classes):
        mask = (y_data == class_idx)
        if mask.sum() > 0:
            steering_vectors[class_idx] = X_data[mask].mean(axis=0)

    return steering_vectors


# ---------------------------------------------------------------------------
# Probing core
# ---------------------------------------------------------------------------

def train_probe_per_layer_timestep(
    train_dir: Path,
    test_dir: Path,
    speakers_list: List[str],
    timesteps_to_probe: List[int],
    layers_to_probe: List[str],
    max_length_sec: float,
    num_heads: int,
    head_dim: int,
    output_dir: Optional[Path] = None,
) -> Tuple[Dict, Dict, Dict, Dict]:
    
    num_speakers = len(speakers_list)

    full_results: Dict[str, Dict[int, Dict]] = defaultdict(dict)
    head_results: Dict[str, Dict[int, Dict[int, Dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    steering_vectors_full: Dict[str, np.ndarray] = {}
    steering_vectors_heads: Dict[str, Dict[int, np.ndarray]] = defaultdict(dict)

    total_experiments = len(layers_to_probe) * len(timesteps_to_probe)
    experiment_count = 0

    for layer_name in layers_to_probe:
        print("\n" + "=" * 70)
        print(f"Probing layer: {layer_name}")
        print("=" * 70)

        is_attention_layer = ".attn" in layer_name and "norm" not in layer_name

        for timestep in timesteps_to_probe:
            experiment_count += 1
            print(
                f"\n[{experiment_count}/{total_experiments}] "
                f"Layer: {layer_name}, Timestep: {timestep}"
            )

            X_train, y_train, _ = load_tensors_from_webdataset_shards(
                shard_dir=train_dir,
                speakers_list=speakers_list,
                target_step=timestep,
                target_layer=layer_name,
                max_length_sec=max_length_sec,
            )

            X_test, y_test, _ = load_tensors_from_webdataset_shards(
                shard_dir=test_dir,
                speakers_list=speakers_list,
                target_step=timestep,
                target_layer=layer_name,
                max_length_sec=max_length_sec,
            )

            if X_train is None or X_test is None:
                print("  No data found, skipping.")
                continue

            feature_dim = X_train.shape[1]

            # Full-layer probe
            probe = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=500,
                    solver="lbfgs",
                    multi_class="multinomial",
                    random_state=42,
                    n_jobs=-1,
                ),
            )

            try:
                probe.fit(X_train, y_train)
                train_accuracy = probe.score(X_train, y_train)
                test_accuracy = probe.score(X_test, y_test)

                steering_vecs_full = compute_steering_vectors(
                    X_train, y_train, num_speakers
                )

                print(
                    f"  Full layer probe: "
                    f"train_acc={train_accuracy:.4f}, test_acc={test_accuracy:.4f}"
                )

                full_results[layer_name][timestep] = {
                    "train_accuracy": float(train_accuracy),
                    "test_accuracy": float(test_accuracy),
                    "feature_dim": int(feature_dim),
                }

                key_full = f"{layer_name}_step{timestep:04d}"
                steering_vectors_full[key_full] = steering_vecs_full

            except Exception as e:
                print(f"  Error training full-layer probe: {e}")
                continue

            # Per-head probes
            if is_attention_layer:
                if feature_dim % num_heads != 0:
                    print(f"  Warning: feature_dim={feature_dim} not divisible by heads={num_heads}")
                    continue

                inferred_head_dim = feature_dim // num_heads
                head_dim_used = inferred_head_dim

                X_train_heads = X_train.reshape(-1, num_heads, head_dim_used)
                X_test_heads = X_test.reshape(-1, num_heads, head_dim_used)

                steering_vecs_per_head = np.zeros(
                    (num_heads, num_speakers, head_dim_used),
                    dtype=np.float32,
                )

                for head_idx in range(num_heads):
                    X_train_h = X_train_heads[:, head_idx, :]
                    X_test_h = X_test_heads[:, head_idx, :]

                    probe_h = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            max_iter=500,
                            solver="lbfgs",
                            multi_class="multinomial",
                            random_state=42,
                            n_jobs=-1,
                        ),
                    )

                    try:
                        probe_h.fit(X_train_h, y_train)
                        train_acc_h = probe_h.score(X_train_h, y_train)
                        test_acc_h = probe_h.score(X_test_h, y_test)

                        steering_vecs_h = compute_steering_vectors(
                            X_train_h, y_train, num_speakers
                        )

                        head_results[layer_name][timestep][head_idx] = {
                            "train_accuracy": float(train_acc_h),
                            "test_accuracy": float(test_acc_h),
                        }

                        steering_vecs_per_head[head_idx] = steering_vecs_h

                    except Exception:
                        continue

                steering_vectors_heads[layer_name][timestep] = steering_vecs_per_head

    # Save results
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # (Saving logic omitted for brevity, identical to previous script)
        # You can copy the DataFrame creation and saving logic here if needed.
        
        steering_full_path = output_dir / "steering_vectors_full.pt"
        torch.save(steering_vectors_full, steering_full_path)

        steering_heads_path = output_dir / "steering_vectors_heads.pt"
        torch.save(steering_vectors_heads, steering_heads_path)

    return full_results, head_results, steering_vectors_full, steering_vectors_heads


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 80)
    print("SPEAKER PROBING: FLAT STRUCTURE")
    print("=" * 80 + "\n")

    BASE_DIR = Path("./extracted_features_sharded")
    TRAIN_TARS_DIR = BASE_DIR / "train_tars"
    TEST_TARS_DIR = BASE_DIR / "test_tars"
    OUTPUT_DIR = BASE_DIR / "probe_results"

    LIBRITTS_ROOT = Path("/scratch/st-jzhu71-1/ewong25/links/scratch/LibriTTS")
    SUBSET_TO_PROCESS = "test-clean"

    MAX_LENGTH_SEC = 5.0
    TIMESTEPS_TO_PROBE = list(range(0, 33, 4))

    MODEL_NAME = "F5TTS_v1_Base"
    LOCAL_MODEL_PATH = Path(
        "/scratch/st-jzhu71-1/ewong25/checkpoints/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    )

    subset_dir = LIBRITTS_ROOT / SUBSET_TO_PROCESS
    if not subset_dir.exists():
        print(f"ERROR: Subset directory {subset_dir} not found.")
        return

    speakers = sorted([p.name for p in subset_dir.iterdir() if p.is_dir()])
    if not speakers:
        print("ERROR: No speakers found.")
        return

    if not TRAIN_TARS_DIR.exists():
        print("ERROR: Train shards not found.")
        return

    print("Inferring attention config...")
    num_heads, head_dim = infer_attention_head_config(
        model_name=MODEL_NAME,
        checkpoint_path=LOCAL_MODEL_PATH,
        device="cpu",
    )

    print("\nDiscovering layers...")
    train_shards = list(TRAIN_TARS_DIR.glob("*.tar"))
    if not train_shards:
        print("No shards found.")
        return
    
    unique_layers = set()
    try:
        dataset = wds.WebDataset(str(train_shards[0]))
        for sample in dataset:
            metadata = json.loads(sample["json"].decode("utf-8"))
            if "layer_group" in metadata:
                unique_layers.add(metadata["layer_group"])
    except Exception as e:
        print(f"Error reading shard: {e}")
        return

    layers_to_probe = sorted(unique_layers)
    print(f"Layers found: {len(layers_to_probe)}")
    for l in layers_to_probe:
        print(f" - {l}")

    train_probe_per_layer_timestep(
        train_dir=TRAIN_TARS_DIR,
        test_dir=TEST_TARS_DIR,
        speakers_list=speakers,
        timesteps_to_probe=TIMESTEPS_TO_PROBE,
        layers_to_probe=layers_to_probe,
        max_length_sec=MAX_LENGTH_SEC,
        num_heads=num_heads,
        head_dim=head_dim,
        output_dir=OUTPUT_DIR,
    )

if __name__ == "__main__":
    main()
