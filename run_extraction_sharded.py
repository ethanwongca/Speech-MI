"""
DiT Activation Extraction - WebDataset Pipeline

This script:
- Runs F5-TTS with ODE sampling.
- Hooks a selected set of DiT transformer blocks (and optionally their main
  submodules) instead of every tiny submodule.
- Captures activations at each ODE timestep via the existing HookConfig/HookManager.
- For each speaker and timestep, writes a single .tar WebDataset shard.
  Each shard contains:
    - one sample per hooked block/submodule
    - "<key>.pt"  : activation tensor bytes
    - "<key>.txt" : "speaker_id timestep block_path" string
"""

from __future__ import annotations

from pathlib import Path
import json
import shutil
from typing import Dict, List, Tuple

import torch
import webdataset
from omegaconf import OmegaConf
from hydra.utils import get_class
from importlib.resources import files

# F5-TTS imports
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    infer_process,
)
from f5_tts.model.cfm import HookConfig, HookManager, CFM
from f5_tts.model.backbones.dit import DiT


# Model / Vocoder loading

def load_f5_model_and_vocoder(device: str = "cuda") -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Load the F5-TTS model and vocoder using local disk paths.

    Args:
        device: "cuda" or "cpu".

    Returns:
        (model, vocoder)
    """
    print(f"Device: {device}")

    model_name = "F5TTS_v1_Base"
    LOCAL_MODEL_PATH = Path(
        "/home/ewong/links/scratch/checkpoints/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    )
    LOCAL_VOCODER_PATH = Path(
        "/home/ewong/links/scratch/checkpoints/vocos-mel-24khz"
    )

    if not LOCAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {LOCAL_MODEL_PATH}")
    if not LOCAL_VOCODER_PATH.exists():
        raise FileNotFoundError(f"Vocoder not found at {LOCAL_VOCODER_PATH}")

    print("Loading vocoder...")
    vocoder = load_vocoder(
        vocoder_name="vocos",
        is_local=True,
        local_path=str(LOCAL_VOCODER_PATH),
        device=device,
    )
    print("Vocoder loaded\n")

    print("Loading F5-TTS model...")
    model_cfg = OmegaConf.load(
        str(files("f5_tts").joinpath(f"configs/{model_name}.yaml"))
    )
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arch = model_cfg.model.arch

    model = load_model(
        model_cls,
        model_arch,
        str(LOCAL_MODEL_PATH),
        mel_spec_type="vocos",
        vocab_file="",
        device=device,
    )
    print("Model loaded\n")

    if not hasattr(model, "transformer"):
        raise AttributeError("Model has no 'transformer' attribute; DiT not found.")

    return model, vocoder


# Hook layer selection utilities

def select_dit_hook_layers(transformer: torch.nn.Module) -> Dict[str, str]:
    """
    Select only the high-level DiT components:

    - time_embed
    - text_embed
    - input_embed
    - rotary_embed
    - norm_out
    - proj_out

    - transformer_blocks.{i}
    - transformer_blocks.{i}.attn_norm
    - transformer_blocks.{i}.attn
    - transformer_blocks.{i}.ff_norm
    - transformer_blocks.{i}.ff

    NOTHING deeper than one level inside these.
    """

    hook_layers = {}

    for module_path, _ in transformer.named_modules():

        # 1. Top-level embeddings and projections
        if module_path in {
            "time_embed",
            "text_embed",
            "input_embed",
            "rotary_embed",
            "norm_out",
            "proj_out",
        }:
            hook_layers[module_path] = module_path
            continue

        # 2. DiT blocks: transformer_blocks.{i}
        if module_path.startswith("transformer_blocks."):
            parts = module_path.split(".")

            # Block-level capture
            if len(parts) == 2 and parts[1].isdigit():
                hook_layers[module_path] = module_path
                continue

            # Capture only these submodules inside a block
            if (
                len(parts) == 3
                and parts[1].isdigit()
                and parts[2] in {"attn_norm", "attn", "ff_norm", "ff"}
            ):
                hook_layers[module_path] = module_path
                continue

            # Otherwise skip deeper modules (e.g. attn.to_q)
            continue

    print(f"[Layer Selection] Selected {len(hook_layers)} capture modules:")
    for name in hook_layers:
        print("  -", name)

    return hook_layers


# WebDataset Writing

def sanitize_block_name(block_name: str) -> str:
    """
    Sanitize a module path so it can be safely used inside a WebDataset key.

    Example:
        "transformer_blocks.0.attn" -> "transformer_blocks_0_attn"
    """
    return block_name.replace("/", "_").replace(".", "_")


def write_step_to_tar(
    speaker_id: str,
    step_index: int,
    step_dir: Path,
    group_names: List[str],
    output_dir: Path,
    split: str,
) -> None:
    """
    For a given speaker and ODE timestep, write a single .tar file
    containing one sample per hooked block/submodule.

    Each sample in the tar has:
      - "<key>.pt"  : activations tensor bytes (as saved by HookConfig)
      - "<key>.txt" : "speaker_id timestep block_path" string

    Args:
        speaker_id: LibriTTS speaker ID (e.g., "1234").
        step_index: ODE timestep index (0-based).
        step_dir: Directory where HookConfig saved grouped tensors for this timestep.
        group_names: List of group names (usually hook_names) for this timestep.
        output_dir: Base directory where .tar shards are written.
        split: "train" or "test".
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tar_path = output_dir / f"speaker_{speaker_id}_step_{step_index:04d}.tar"
    print(f"  Writing shard: {tar_path}")

    # One tar per (speaker, timestep)
    with webdataset.TarWriter(str(tar_path)) as writer:
        for group_name in group_names:
            safe_filename = f"{sanitize_block_name(group_name)}.pt"
            tensor_path = step_dir / safe_filename

            if not tensor_path.exists():
                print(f"    Warning: tensor file {tensor_path} not found, skipping.")
                continue

            tensor_bytes = tensor_path.read_bytes()

            # Construct a unique key for this sample inside the tar
            # Example: "1234_t0003_b_transformer_blocks_0_attn"
            key = f"{speaker_id}_t{step_index:04d}_b{sanitize_block_name(group_name)}"

            # Text sidecar: "speaker timestep block_path split"
            txt_content = f"{speaker_id} {step_index} {group_name} {split}"
            txt_bytes = txt_content.encode("utf-8")

            sample = {
                "__key__": key,
                "pt": tensor_bytes,
                "txt": txt_bytes,
            }
            writer.write(sample)


def pack_speaker_from_temp(
    speaker_id: str,
    temp_dir_base: Path,
    output_dir: Path,
    split: str,
) -> bool:
    """
    Read the HookConfig metadata and per-timestep tensor groups for a speaker,
    and pack them into WebDataset shards.

    For each timestep, this writes exactly one .tar:
      "speaker_{speaker_id}_step_{step:04d}.tar"

    Args:
        speaker_id: LibriTTS speaker ID.
        temp_dir_base: Base directory for temporary saved activations, e.g. "./tmp/train".
        output_dir: Directory for final .tar shards for this split.
        split: "train" or "test".

    Returns:
        True if at least one shard was written, False otherwise.
    """
    run_temp_dir = temp_dir_base / speaker_id
    metadata_path = run_temp_dir / "capture_metadata.json"

    if not metadata_path.exists():
        print(f"  No metadata found at {metadata_path}, nothing to pack.")
        return False

    with open(metadata_path, "r") as f:
        meta = json.load(f)

    timesteps = meta.get("timesteps", [])
    if not timesteps:
        print("  No timesteps recorded in metadata.")
        return False

    print(f"  Packing {len(timesteps)} timesteps for speaker {speaker_id} → WebDataset.")

    shards_written = 0
    for ts_info in timesteps:
        step_index = ts_info.get("step_index")
        step_dir = Path(ts_info.get("directory", ""))
        group_names = ts_info.get("group_names", [])

        if not step_dir.exists():
            print(f"  Warning: timestep directory {step_dir} missing, skipping.")
            continue

        if not group_names:
            print(f"  Warning: no group_names listed for step {step_index}, skipping.")
            continue

        write_step_to_tar(
            speaker_id=speaker_id,
            step_index=step_index,
            step_dir=step_dir,
            group_names=group_names,
            output_dir=output_dir,
            split=split,
        )
        shards_written += 1

    # Optional: cleanup temp directory for this speaker
    if run_temp_dir.exists():
        shutil.rmtree(run_temp_dir, ignore_errors=True)

    print(f"  Finished packing: wrote {shards_written} shards.")
    return shards_written > 0


# Capture (Hooks + ODE sampling)

def run_speaker_capture(
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    speaker_id: str,
    wav_path: Path,
    text_prompt: str,
    temp_dir_base: Path,
    inference_params: dict,
    device: str,
    split_label: str,
) -> bool:
    """
    Run one ODE inference pass for a speaker and capture activations
    at selected DiT layers for each ODE timestep.

    The HookConfig/HookManager from f5_tts.model.cfm handle:
      - attaching forward hooks
      - capturing activations at each ODE solver step
      - writing grouped tensors and metadata into temp_dir_base / speaker_id

    Args:
        model: F5-TTS model (top-level CFM wrapper).
        vocoder: Vocos/vocoder model.
        speaker_id: LibriTTS speaker ID.
        wav_path: Path to reference waveform.
        text_prompt: Associated normalized text.
        temp_dir_base: Base directory for temporary capture files.
        inference_params: Dict of arguments passed through to infer_process.
        device: "cuda" or "cpu".
        split_label: "train" or "test" (for logging only).

    Returns:
        True if capture completed without raising, False otherwise.
    """
    print(f"\n--- CAPTURE ({split_label}) for Speaker {speaker_id} ({wav_path.name}) ---")

    run_temp_dir = temp_dir_base / speaker_id
    run_temp_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(model, "transformer"):
        raise AttributeError("Model has no 'transformer' attribute; DiT not found.")

    transformer = model.transformer

    # Preprocess reference audio/text once per speaker
    try:
        ref_audio, ref_text = preprocess_ref_audio_text(str(wav_path), text_prompt)
    except Exception as e:
        print(f"  Preprocessing failed for speaker {speaker_id}: {e}")
        return False

    # Select which DiT modules to hook
    capture_layers = select_dit_hook_layers(transformer)

    if not capture_layers:
        print("  No layers selected for capture; aborting.")
        return False

    # Create HookConfig + HookManager (ODE will call HookManager per timestep)
    hook_config = HookConfig(
        capture_all_submodules=False,
        capture_all_layers=False,
        capture_layers=capture_layers,
        capture_final_output=False,
        time_filter=lambda t, step: True,   # capture at all timesteps
        precision=torch.bfloat16,
        device=device,
        save_per_timestep=True,             # write per-step grouped tensors to disk
        save_dir=str(run_temp_dir),
        patches=None,
        clear_activations_per_step=True,    # free activations after each step
    )

    hook_manager = HookManager(hook_config)

    # Run inference once, with hooks attached
    try:
        _ = infer_process(
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=ref_text,
            model_obj=model,
            vocoder=vocoder,
            hook_manager=hook_manager,
            **inference_params,
        )
    except Exception as e:
        print(f"  Capture failed for speaker {speaker_id}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Ensure hooks are removed even if inference fails
        if hasattr(transformer, "remove_hooks"):
            transformer.remove_hooks()
        if hasattr(transformer, "clear_activations"):
            transformer.clear_activations()

    print(f"  Capture complete. Temporary files in {run_temp_dir}")
    return True


# Main driver

def main() -> None:
    """
    Top-level dataset builder.

    For each speaker in a LibriTTS subset:
      1. Capture activations (train utterance + test utterance) at selected DiT layers.
      2. Convert captures into WebDataset shards:
         - one shard (.tar) per (speaker, timestep) and split.

    The output directory structure is:

        extracted_features_sharded/
          train_tars/
            speaker_XXXX_step_0000.tar
            ...
          test_tars/
            speaker_XXXX_step_0000.tar
            ...
    """
    print("\n" + "=" * 80)
    print("DiT ACTIVATION EXTRACTION → WEBDATASET")
    print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # Paths and configuration
    # ------------------------------------------------------------------
    LIBRITTS_ROOT = Path("/home/ewong/links/scratch/LibriTTS")
    SUBSET_TO_PROCESS = "test-clean"  # e.g., "train-clean-100"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    OUTPUT_DIR = Path("./extracted_features_sharded")
    TRAIN_TARS_DIR = OUTPUT_DIR / "train_tars"
    TEST_TARS_DIR = OUTPUT_DIR / "test_tars"
    TEMP_DIR = OUTPUT_DIR / "tmp"

    TRAIN_TARS_DIR.mkdir(parents=True, exist_ok=True)
    TEST_TARS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Load model + vocoder
    try:
        model, vocoder = load_f5_model_and_vocoder(device=DEVICE)
    except Exception as e:
        print(f"Fatal error loading model/vocoder: {e}")
        return

    # Inference parameters passed to infer_process
    inference_params = {
        "mel_spec_type": "vocos",
        "target_rms": 0.1,
        "cross_fade_duration": 0.15,
        "nfe_step": 32,
        "cfg_strength": 2.0,
        "sway_sampling_coef": -1,
        "speed": 1.0,
        "fix_duration": None,
        "device": DEVICE,
    }

    # ------------------------------------------------------------------
    # Discover speakers in the chosen LibriTTS subset
    # ------------------------------------------------------------------
    subset_dir = LIBRITTS_ROOT / SUBSET_TO_PROCESS
    if not subset_dir.exists():
        print(f"ERROR: Subset directory {subset_dir} does not exist.")
        return

    speaker_dirs = sorted([p for p in subset_dir.iterdir() if p.is_dir()])
    print(f"Found {len(speaker_dirs)} speakers in subset '{SUBSET_TO_PROCESS}'.")

    if not speaker_dirs:
        print("No speakers found; exiting.")
        return

    successful_count = 0
    failed_count = 0

    for idx, speaker_dir in enumerate(speaker_dirs):
        speaker_id = speaker_dir.name
        print(
            f"\n{'=' * 80}\n"
            f"Processing Speaker {idx + 1}/{len(speaker_dirs)}: {speaker_id}\n"
            f"{'=' * 80}"
        )

        wav_files = sorted(list(speaker_dir.rglob("*.wav")))
        if len(wav_files) < 2:
            print(
                f"  Skipping speaker {speaker_id}: need at least 2 WAV files "
                f"(found {len(wav_files)})."
            )
            failed_count += 1
            continue

        # Use first file for "train" and second for "test" for this script
        train_wav = wav_files[0]
        test_wav = wav_files[1]
        train_txt = train_wav.with_suffix(".normalized.txt")
        test_txt = test_wav.with_suffix(".normalized.txt")

        if not (train_txt.exists() and test_txt.exists()):
            print(
                f"  Skipping speaker {speaker_id}: missing normalized text "
                f"({train_txt}, {test_txt})."
            )
            failed_count += 1
            continue

        try:
            train_text = train_txt.read_text().strip()
            test_text = test_txt.read_text().strip()
        except Exception as e:
            print(f"  Skipping speaker {speaker_id}: error reading text files: {e}")
            failed_count += 1
            continue

        speaker_ok = True

        # ------------------------------------------------------------------
        # TRAIN split: capture → pack
        # ------------------------------------------------------------------
        print("\n" + "-" * 80)
        print(f"TRAIN split for speaker {speaker_id}")
        print("-" * 80)

        if run_speaker_capture(
            model=model,
            vocoder=vocoder,
            speaker_id=speaker_id,
            wav_path=train_wav,
            text_prompt=train_text,
            temp_dir_base=TEMP_DIR / "train",
            inference_params=inference_params,
            device=DEVICE,
            split_label="train",
        ):
            if not pack_speaker_from_temp(
                speaker_id=speaker_id,
                temp_dir_base=TEMP_DIR / "train",
                output_dir=TRAIN_TARS_DIR,
                split="train",
            ):
                speaker_ok = False
        else:
            speaker_ok = False

        # ------------------------------------------------------------------
        # TEST split: capture → pack
        # ------------------------------------------------------------------
        print("\n" + "-" * 80)
        print(f"TEST split for speaker {speaker_id}")
        print("-" * 80)

        if run_speaker_capture(
            model=model,
            vocoder=vocoder,
            speaker_id=speaker_id,
            wav_path=test_wav,
            text_prompt=test_text,
            temp_dir_base=TEMP_DIR / "test",
            inference_params=inference_params,
            device=DEVICE,
            split_label="test",
        ):
            if not pack_speaker_from_temp(
                speaker_id=speaker_id,
                temp_dir_base=TEMP_DIR / "test",
                output_dir=TEST_TARS_DIR,
                split="test",
            ):
                speaker_ok = False
        else:
            speaker_ok = False

        if speaker_ok:
            successful_count += 1
        else:
            failed_count += 1

    # Final summary
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"  Train shards: {TRAIN_TARS_DIR}")
    print(f"  Test shards : {TEST_TARS_DIR}")
    print(f"Successfully processed: {successful_count}")
    print(f"Failed: {failed_count}")

    # Optional final cleanup of any remaining temp files
    if TEMP_DIR.exists():
        try:
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
            print("Temporary directory cleaned up.")
        except Exception as e:
            print(f"Warning: could not fully clean TEMP_DIR: {e}")


if __name__ == "__main__":
    main()
