#!/usr/bin/env python3
"""
Speaker Probing over DiT Hidden States and Attention Heads
==========================================================

Pipeline overview
-----------------
This script operates on WebDataset shards produced by your extraction pipeline
(e.g. `extracted_features_sharded/train_tars` and `.../test_tars`), where each
shard contains activations for a given speaker, timestep, and layer group.

Assumed WebDataset format
-------------------------
Each sample in a shard is expected to have:
  - "pth": bytes of a torch.save'd dict with structure:
        {
            "t": float,
            "step_index": int,
            "group_name": str,            # e.g. "block0_attn"
            "activations": {
                "__output__": tensor,     # shape [B, T, D] or [T, D] or similar
                ...                      # (we use "__output__" if present)
            }
        }
    If "activations" is absent, we treat the loaded object as the tensor.

  - "json": UTF-8 encoded metadata JSON:
        {
            "speaker_id": str,
            "step_index": int,
            "layer_group": str,           # e.g. "block0_attn"
            "split": "train" | "test"
        }

Naming convention for layers
----------------------------
The script expects layer group names of the form:

  - "input_embed"
  - "block{i}"
  - "block{i}_attn"
  - "block{i}_ff"
  - "norm_out"
  - "proj_out"

for i in [0, ..., depth-1].

Probing setup
-------------
For each layer and timestep:
  - Load hidden states from train shards (one utterance per speaker).
  - Truncate to a maximum of `max_length_sec` seconds (using frames_per_sec).
  - Flatten over time, giving N_samples x D feature matrix.
  - Train a multinomial Logistic Regression probe to classify speakers.
  - Evaluate train and test accuracy.
  - Compute steering vectors (class centroids) per layer/timestep.

For attention layers (names ending with "_attn"):
  - Derive num_heads and head_dim from the DiT model configuration.
  - Reshape features [N, D] into [N, num_heads, head_dim].
  - For each head, train a separate logistic regression and compute steering
    vectors per head.

Outputs
-------
- probe_train_accuracy.csv          : full-layer train accuracy per layer/timestep
- probe_test_accuracy.csv           : full-layer test accuracy per layer/timestep
- probe_head_accuracy.csv           : per-head test/train accuracy per layer/head/timestep
- steering_vectors_full.pt          : steering vectors for full layers
- steering_vectors_heads.pt         : steering vectors for attention heads
- probe_accuracy_heatmaps.png       : heatmaps for full-layer accuracies

You can customize the list of layers/timesteps to probe and other hyperparameters
in the `main()` function.
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


# Optional: Only needed to infer attention head configuration from model
from omegaconf import OmegaConf
from hydra.utils import get_class
from importlib.resources import files
from f5_tts.infer.utils_infer import load_model


# ---------------------------------------------------------------------------
# Model utilities: infer attention head configuration from DiT
# ---------------------------------------------------------------------------

def load_dit_model_for_config(
    model_name: str,
    checkpoint_path: Path,
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Load the F5-TTS model just to access the DiT backbone and its attention config.

    This function loads the model on CPU by default, since we only need its
    architecture to infer number of attention heads and head dimension.

    Args:
        model_name: Name of the F5-TTS config (e.g. "F5TTS_v1_Base").
        checkpoint_path: Path to the model checkpoint.
        device: Device to load the model on ("cpu" recommended).

    Returns:
        dit_model: The DiT transformer module.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    model_cfg = OmegaConf.load(
        str(files("f5_tts").joinpath(f"configs/{model_name}.yaml"))
    )
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arch = model_cfg.model.arch

    # We do not need the vocoder here, only the backbone model
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
    """
    Infer the number of attention heads and head dimension from the DiT config.

    This inspects the first transformer block's attention module and assumes
    all attention layers share the same (num_heads, head_dim).

    Args:
        model_name: F5-TTS config name (e.g., "F5TTS_v1_Base").
        checkpoint_path: Path to the model checkpoint.
        device: Device to load the model on.

    Returns:
        (num_heads, head_dim)
    """
    dit_model = load_dit_model_for_config(model_name, checkpoint_path, device=device)

    if not hasattr(dit_model, "transformer_blocks"):
        raise AttributeError("DiT model has no 'transformer_blocks' attribute.")

    if len(dit_model.transformer_blocks) == 0:
        raise ValueError("DiT model has zero transformer blocks.")

    attn_module = dit_model.transformer_blocks[0].attn

    if not hasattr(attn_module, "heads"):
        raise AttributeError("Attention module does not expose 'heads' attribute.")

    num_heads = int(attn_module.heads)

    # x-transformers Attention usually exposes dim_head; fall back to dim/heads if needed
    if hasattr(attn_module, "dim_head"):
        head_dim = int(attn_module.dim_head)
    else:
        if hasattr(attn_module, "dim"):
            hidden_dim = int(attn_module.dim)
        else:
            raise AttributeError(
                "Attention module missing both 'dim_head' and 'dim'; "
                "cannot infer head_dim."
            )
        head_dim = hidden_dim // num_heads

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
    Load hidden states from WebDataset shards for a specific timestep and layer.

    Each shard is assumed to store per-layer activations for a particular speaker
    and timestep in the format produced by the extraction pipeline.

    For each speaker and shard:
        - Load the tensor for the requested layer and timestep.
        - Truncate to at most max_frames.
        - Flatten over time and accumulate samples with speaker labels.

    Args:
        shard_dir:
            Directory containing shard files (e.g. train_tars or test_tars).
        speakers_list:
            List of speaker IDs to consider (all speakers in LibriTTS subset).
        target_step:
            ODE timestep index to load (int).
        target_layer:
            Layer group name, e.g. "block0_attn", "block0", "input_embed".
        max_length_sec:
            Maximum number of seconds to keep (converted to frames with frames_per_sec).
        frames_per_sec:
            Frames per second for the underlying representation (e.g., mel frames).

    Returns:
        X_data:
            N x D array of features (float32) or None if no data.
        y_data:
            N array of integer speaker labels or None.
        speaker_boundaries:
            List of (start_index, end_index) tuples indicating which rows
            belong to each speaker in speakers_list. None if no data.
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
            # No data for this speaker and timestep
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
                        if "__output__" in activations:
                            tensor = activations["__output__"]
                        else:
                            tensor = list(activations.values())[0]
                    else:
                        tensor = data_dict

                    # Normalize shapes to (T, D) or (B*T, D)
                    if tensor.ndim == 3:
                        # (batch, seq, features)
                        # For this extraction setup, batch is usually 1 or 2.
                        tensor = tensor.reshape(-1, tensor.shape[-1])
                    elif tensor.ndim == 2:
                        # (seq, features)
                        pass
                    else:
                        # Unexpected shape, skip
                        continue

                    # Truncate to max_frames
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
    """
    Compute steering vectors (class centroids) for each class.

    Given feature matrix X_data (N x D) and labels y_data (N), compute the
    mean vector for each class index in [0, num_classes).

    Args:
        X_data:
            N x D feature array.
        y_data:
            Length-N array of integer class labels.
        num_classes:
            Number of distinct classes.

    Returns:
        steering_vectors:
            num_classes x D array, where steering_vectors[c] is the centroid
            for class c (or all zeros if class c is not present).
    """
    feature_dim = X_data.shape[1]
    steering_vectors = np.zeros((num_classes, feature_dim), dtype=np.float32)

    for class_idx in range(num_classes):
        mask = (y_data == class_idx)
        if mask.sum() > 0:
            steering_vectors[class_idx] = X_data[mask].mean(axis=0)

    return steering_vectors


# ---------------------------------------------------------------------------
# Probing core: per layer, per timestep, plus per-head on attention layers
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
) -> Tuple[
    Dict[str, Dict[int, Dict]],
    Dict[str, Dict[int, Dict[int, Dict]]],
    Dict[str, np.ndarray],
    Dict[str, Dict[int, np.ndarray]],
]:
    """
    Train logistic regression probes for each layer and timestep, plus per-head
    probes for attention layers.

    For each layer in layers_to_probe and each timestep in timesteps_to_probe:
      - Load train and test hidden states from WebDataset shards.
      - Train a multinomial logistic regression probe on the train data.
      - Evaluate on both train and test.
      - Compute steering vectors on train features.

    For layers whose name ends with "_attn", also:
      - Reshape features to [N, num_heads, head_dim].
      - For each head index h in [0, num_heads):
          - Train a separate probe on the h-th head's features.
          - Compute steering vectors in head space.

    Args:
        train_dir:
            Directory containing training shards.
        test_dir:
            Directory containing test shards.
        speakers_list:
            List of speaker IDs (classes).
        timesteps_to_probe:
            List of timestep indices to probe.
        layers_to_probe:
            List of layer group names to probe.
        max_length_sec:
            Maximum sequence length in seconds (truncation is done in loading).
        num_heads:
            Number of attention heads (derived from DiT config).
        head_dim:
            Dimension per head (derived from DiT config).
        output_dir:
            Directory to save results (CSVs, heatmaps, steering vectors).

    Returns:
        full_results:
            Nested dict full_results[layer_name][timestep] = metrics dict.
        head_results:
            Nested dict head_results[layer_name][timestep][head_idx] = metrics.
        steering_vectors_full:
            Mapping from "layer_step" → steering vector array (num_speakers x D).
        steering_vectors_heads:
            Mapping from layer_name → {timestep → steering vectors per head}
            where steering_vectors_heads[layer][timestep] has shape
            (num_heads, num_speakers, head_dim).
    """
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

        is_attention_layer = layer_name.endswith("_attn")

        for timestep in timesteps_to_probe:
            experiment_count += 1
            print(
                f"\n[{experiment_count}/{total_experiments}] "
                f"Layer: {layer_name}, Timestep: {timestep}"
            )

            # Load training data
            print("  Loading train data...")
            X_train, y_train, train_boundaries = load_tensors_from_webdataset_shards(
                shard_dir=train_dir,
                speakers_list=speakers_list,
                target_step=timestep,
                target_layer=layer_name,
                max_length_sec=max_length_sec,
            )

            # Load test data
            print("  Loading test data...")
            X_test, y_test, test_boundaries = load_tensors_from_webdataset_shards(
                shard_dir=test_dir,
                speakers_list=speakers_list,
                target_step=timestep,
                target_layer=layer_name,
                max_length_sec=max_length_sec,
            )

            if X_train is None or X_test is None:
                print("  No data found for this layer/timestep, skipping.")
                continue

            print(f"  Train features: {X_train.shape}, Test features: {X_test.shape}")

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
                    "num_train_samples": int(X_train.shape[0]),
                    "num_test_samples": int(X_test.shape[0]),
                    "feature_dim": int(feature_dim),
                }

                key_full = f"{layer_name}_step{timestep:04d}"
                steering_vectors_full[key_full] = steering_vecs_full

            except Exception as e:
                print(f"  Error training full-layer probe: {e}")
                continue

            # Per-head probes for attention layers
            if is_attention_layer:
                if feature_dim % num_heads != 0:
                    print(
                        f"  Warning: feature_dim={feature_dim} is not divisible by "
                        f"num_heads={num_heads}. Skipping per-head probes."
                    )
                    continue

                inferred_head_dim = feature_dim // num_heads
                if inferred_head_dim != head_dim:
                    print(
                        f"  Note: inferred head_dim={inferred_head_dim} from data "
                        f"differs from config head_dim={head_dim}. Using inferred value."
                    )
                    head_dim_used = inferred_head_dim
                else:
                    head_dim_used = head_dim

                # Reshape into [N, num_heads, head_dim]
                X_train_heads = X_train.reshape(-1, num_heads, head_dim_used)
                X_test_heads = X_test.reshape(-1, num_heads, head_dim_used)

                # Steering vectors per head: shape (num_heads, num_speakers, head_dim)
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
                            "num_train_samples": int(X_train_h.shape[0]),
                            "num_test_samples": int(X_test_h.shape[0]),
                            "feature_dim": int(head_dim_used),
                        }

                        steering_vecs_per_head[head_idx] = steering_vecs_h

                        print(
                            f"    Head {head_idx:02d}: "
                            f"train_acc={train_acc_h:.4f}, test_acc={test_acc_h:.4f}"
                        )

                    except Exception as e:
                        print(
                            f"    Error training head {head_idx} probe "
                            f"for {layer_name}, step {timestep}: {e}"
                        )
                        continue

                steering_vectors_heads[layer_name][timestep] = steering_vecs_per_head

    # Save results if requested
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Full-layer accuracy tables
        if full_results:
            # Build DataFrames: rows=layers, columns=timesteps
            train_data = {}
            test_data = {}

            for layer, steps in full_results.items():
                train_data[layer] = {}
                test_data[layer] = {}
                for step, metrics in steps.items():
                    train_data[layer][step] = metrics["train_accuracy"]
                    test_data[layer][step] = metrics["test_accuracy"]

            df_train = pd.DataFrame(train_data).T
            df_test = pd.DataFrame(test_data).T

            df_train = df_train.reindex(sorted(df_train.columns), axis=1)
            df_test = df_test.reindex(sorted(df_test.columns), axis=1)

            train_csv = output_dir / "probe_train_accuracy.csv"
            test_csv = output_dir / "probe_test_accuracy.csv"
            df_train.to_csv(train_csv)
            df_test.to_csv(test_csv)
            print(f"\nSaved full-layer train accuracies to {train_csv}")
            print(f"Saved full-layer test accuracies to {test_csv}")

            # Heatmaps for full-layer probes
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(20, max(10, len(layers_to_probe) * 0.3)),
            )

            sns.heatmap(
                df_train,
                annot=True,
                fmt=".3f",
                cmap="viridis",
                cbar_kws={"label": "Train Accuracy"},
                vmin=0.0,
                vmax=1.0,
                ax=axes[0],
            )
            axes[0].set_title("Train Accuracy (full layer)")
            axes[0].set_xlabel("ODE timestep")
            axes[0].set_ylabel("Layer")

            sns.heatmap(
                df_test,
                annot=True,
                fmt=".3f",
                cmap="viridis",
                cbar_kws={"label": "Test Accuracy"},
                vmin=0.0,
                vmax=1.0,
                ax=axes[1],
            )
            axes[1].set_title("Test Accuracy (full layer)")
            axes[1].set_xlabel("ODE timestep")
            axes[1].set_ylabel("Layer")

            plt.tight_layout()
            heatmap_path = output_dir / "probe_accuracy_heatmaps.png"
            plt.savefig(heatmap_path, dpi=150)
            plt.close(fig)
            print(f"Saved full-layer accuracy heatmaps to {heatmap_path}")

        # Per-head accuracy table
        if head_results:
            head_records = []
            for layer, steps in head_results.items():
                for step, head_dict in steps.items():
                    for head_idx, metrics in head_dict.items():
                        head_records.append(
                            {
                                "layer": layer,
                                "head": head_idx,
                                "timestep": step,
                                "train_accuracy": metrics["train_accuracy"],
                                "test_accuracy": metrics["test_accuracy"],
                            }
                        )

            df_heads = pd.DataFrame(head_records)
            head_csv = output_dir / "probe_head_accuracy.csv"
            df_heads.to_csv(head_csv, index=False)
            print(f"Saved per-head accuracies to {head_csv}")

        # Steering vectors
        steering_full_path = output_dir / "steering_vectors_full.pt"
        torch.save(steering_vectors_full, steering_full_path)
        print(f"Saved full-layer steering vectors to {steering_full_path}")

        steering_heads_path = output_dir / "steering_vectors_heads.pt"
        torch.save(steering_vectors_heads, steering_heads_path)
        print(f"Saved per-head steering vectors to {steering_heads_path}")

        # Summary
        if full_results:
            print("\n" + "=" * 70)
            print("SUMMARY (full-layer probes)")
            print("=" * 70)
            df_test = pd.read_csv(output_dir / "probe_test_accuracy.csv", index_col=0)
            best_layer = df_test.max(axis=1).idxmax()
            best_layer_acc = df_test.max(axis=1).max()
            best_step = df_test.max(axis=0).idxmax()
            best_step_acc = df_test.max(axis=0).max()
            print(
                f"Best layer (by max test accuracy across timesteps): "
                f"{best_layer} (test_acc={best_layer_acc:.4f})"
            )
            print(
                f"Best timestep (by max test accuracy across layers): "
                f"{best_step} (test_acc={best_step_acc:.4f})"
            )

    return full_results, head_results, steering_vectors_full, steering_vectors_heads


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """
    Main probing pipeline.

    This function:
      1. Locates train and test WebDataset shards.
      2. Derives the attention head configuration from the DiT model.
      3. Discovers available layers from a sample shard.
      4. Trains probes per layer and timestep, including per-head probes for
         attention layers.
      5. Saves results (CSVs, heatmaps, steering vectors) to disk.
    """
    print("\n" + "=" * 80)
    print("SPEAKER PROBING: PER TIMESTEP / LAYER / ATTENTION HEAD")
    print("=" * 80 + "\n")

    # Base paths for extracted features and LibriTTS subset
    BASE_DIR = Path("./extracted_features_sharded")
    TRAIN_TARS_DIR = BASE_DIR / "train_tars"
    TEST_TARS_DIR = BASE_DIR / "test_tars"
    OUTPUT_DIR = BASE_DIR / "probe_results"

    LIBRITTS_ROOT = Path("/home/ewong/links/scratch/LibriTTS")
    SUBSET_TO_PROCESS = "test-clean"

    # Probing configuration
    MAX_LENGTH_SEC = 5.0        # truncate to max 5 seconds
    TIMESTEPS_TO_PROBE = list(range(0, 33, 4))  # every 4th timestep (0, 4, 8, ... 32)

    # Model configuration for inferring attention head structure
    MODEL_NAME = "F5TTS_v1_Base"
    LOCAL_MODEL_PATH = Path(
        "/home/ewong/links/scratch/checkpoints/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    )

    # Discover speakers from LibriTTS subset
    subset_dir = LIBRITTS_ROOT / SUBSET_TO_PROCESS
    if not subset_dir.exists():
        print(f"ERROR: Subset directory {subset_dir} not found.")
        return

    speakers = sorted([p.name for p in subset_dir.iterdir() if p.is_dir()])
    if not speakers:
        print("ERROR: No speakers found in LibriTTS subset.")
        return

    print(f"Found {len(speakers)} speakers in subset {SUBSET_TO_PROCESS}.")

    # Check shard directories
    if not TRAIN_TARS_DIR.exists() or not TEST_TARS_DIR.exists():
        print("ERROR: Train/test shard directories not found. Run extraction script first.")
        return

    train_shards = list(TRAIN_TARS_DIR.glob("speaker_*_step_*.tar"))
    test_shards = list(TEST_TARS_DIR.glob("speaker_*_step_*.tar"))

    if not train_shards or not test_shards:
        print("ERROR: No shard files found. Run extraction script first.")
        return

    print(f"Found {len(train_shards)} train shards and {len(test_shards)} test shards.")

    # Infer attention head configuration from model
    print("\nInferring attention head configuration from DiT model...")
    num_heads, head_dim = infer_attention_head_config(
        model_name=MODEL_NAME,
        checkpoint_path=LOCAL_MODEL_PATH,
        device="cpu",
    )
    print(f"Attention configuration: num_heads={num_heads}, head_dim={head_dim}")

    # Discover available layers from a sample shard
    print("\nDiscovering available layers from a sample shard...")
    sample_shard = train_shards[0]
    unique_layers = set()

    try:
        dataset = wds.WebDataset(str(sample_shard))
        for sample in dataset:
            metadata = json.loads(sample["json"].decode("utf-8"))
            layer_group = metadata.get("layer_group", "")
            if layer_group:
                unique_layers.add(layer_group)
        if not unique_layers:
            print("ERROR: No layer_group entries found in sample shard.")
            return
    except Exception as e:
        print(f"ERROR while inspecting sample shard: {e}")
        return

    layers_to_probe = sorted(unique_layers)
    print(f"Discovered {len(layers_to_probe)} unique layer groups:")
    for name in layers_to_probe:
        print(f"  - {name}")

    # Run probing
    print(
        f"\nTraining probes for {len(layers_to_probe)} layers across "
        f"{len(TIMESTEPS_TO_PROBE)} timesteps..."
    )
    print(f"Maximum sequence length: {MAX_LENGTH_SEC} seconds\n")

    (
        full_results,
        head_results,
        steering_vectors_full,
        steering_vectors_heads,
    ) = train_probe_per_layer_timestep(
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

    print("\n" + "=" * 80)
    print("PROBING COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(
        f"Steering vectors (full layer): "
        f"{len(steering_vectors_full)} layer/timestep combinations"
    )

    num_head_entries = sum(
        1
        for layer_dict in steering_vectors_heads.values()
        for _ in layer_dict.values()
    )
    print(
        f"Steering vectors (per-head): "
        f"{num_head_entries} layer/timestep combinations"
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
