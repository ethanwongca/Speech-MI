"""
DiT Activation Extraction - WebDataset Pipeline (Per-Speaker Sharding + All Hooks)

This script:
- Runs F5-TTS with ODE sampling.
- Hooks ALL DiT transformer blocks and their internal submodules (attn, ff, etc.).
- Captures activations at each ODE timestep.
- Writes a SINGLE .tar WebDataset shard per speaker containing all timesteps.
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
from f5_tts.model.cfm import HookConfig, HookManager


# Model / Vocoder loading

def load_f5_model_and_vocoder(device: str = "cuda") -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Load the F5-TTS model and vocoder using local disk paths."""
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
    Select specific DiT components based on embedding_layer_calls.txt:
    - embeddings: time_embed, text_embed, input_embed
    - blocks & submodules: transformer_blocks.N, .N.attn, .N.ff, etc.
    - output: norm_out, proj_out
    """

    hook_layers = {}

    for module_path, _ in transformer.named_modules():

        # 1. Embeddings and Output
        if module_path in {
            "time_embed", "text_embed", "input_embed", "norm_out", "proj_out",
        }:
            hook_layers[module_path] = module_path
            continue

        # 2. DiT blocks and components
        if module_path.startswith("transformer_blocks."):
            parts = module_path.split(".")

            # Capture the block output itself: transformer_blocks.N
            if len(parts) == 2 and parts[1].isdigit():
                hook_layers[module_path] = module_path
                continue

            # RESTORED: Capture specific sub-components: transformer_blocks.N.{attn, ff, etc}
            if (
                len(parts) == 3
                and parts[1].isdigit()
                and parts[2] in {"attn_norm", "attn", "ff_norm", "ff"}
            ):
                hook_layers[module_path] = module_path
                continue

    print(f"[Layer Selection] Selected {len(hook_layers)} capture modules:")
    # for name in sorted(hook_layers.keys()):
    #     print("  -", name)

    return hook_layers


# WebDataset Writing

def sanitize_block_name(block_name: str) -> str:
    """Sanitize a module path so it can be safely used inside a WebDataset key."""
    return block_name.replace("/", "_").replace(".", "_")


def write_step_to_writer(
    writer: webdataset.TarWriter,
    speaker_id: str,
    step_index: int,
    step_dir: Path,
    group_names: List[str],
    split: str,
) -> None:
    """
    Writes samples for a specific timestep into the provided TarWriter.
    """
    for group_name in group_names:
        safe_filename = f"{sanitize_block_name(group_name)}.pt"
        tensor_path = step_dir / safe_filename

        if not tensor_path.exists():
            continue

        tensor_bytes = tensor_path.read_bytes()

        # Key example: "1234_t0003_b_transformer_blocks_0_attn"
        key = f"{speaker_id}_t{step_index:04d}_b_{sanitize_block_name(group_name)}"

        # Text sidecar: "speaker timestep layer_path split"
        txt_content = f"{speaker_id} {step_index} {group_name} {split}"
        
        # Metadata
        metadata = {
            "speaker_id": speaker_id,
            "step_index": step_index,
            "layer_group": group_name,
            "split": split
        }

        sample = {
            "__key__": key,
            "pth": tensor_bytes,
            "txt": txt_content.encode("utf-8"),
            "json": json.dumps(metadata).encode("utf-8")
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
    and pack them into A SINGLE WebDataset shard per speaker.
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

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create one tar file for the entire speaker
    tar_path = output_dir / f"speaker_{speaker_id}.tar"
    print(f"  Packing {len(timesteps)} timesteps -> {tar_path}")

    shards_written = 0
    
    with webdataset.TarWriter(str(tar_path)) as writer:
        for ts_info in timesteps:
            step_index = ts_info.get("step_index")
            step_dir = Path(ts_info.get("directory", ""))
            group_names = ts_info.get("group_names", [])

            if not step_dir.exists():
                continue

            if not group_names:
                continue

            write_step_to_writer(
                writer=writer,
                speaker_id=speaker_id,
                step_index=step_index,
                step_dir=step_dir,
                group_names=group_names,
                split=split,
            )
            shards_written += 1

    if run_temp_dir.exists():
        shutil.rmtree(run_temp_dir, ignore_errors=True)

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
    """Run one ODE inference pass for a speaker and capture activations."""
    print(f"\n--- CAPTURE ({split_label}) for Speaker {speaker_id} ({wav_path.name}) ---")

    run_temp_dir = temp_dir_base / speaker_id
    run_temp_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(model, "transformer"):
        raise AttributeError("Model has no 'transformer' attribute; DiT not found.")

    transformer = model.transformer

    try:
        ref_audio, ref_text = preprocess_ref_audio_text(str(wav_path), text_prompt)
    except Exception as e:
        print(f"  Preprocessing failed for speaker {speaker_id}: {e}")
        return False

    capture_layers = select_dit_hook_layers(transformer)

    if not capture_layers:
        print("  No layers selected for capture; aborting.")
        return False

    hook_config = HookConfig(
        capture_all_submodules=False,
        capture_all_layers=False,
        capture_layers=capture_layers,
        capture_final_output=False,
        time_filter=lambda t, step: True,
        precision=torch.bfloat16,
        device=device,
        save_per_timestep=True,
        save_dir=str(run_temp_dir),
        patches=None,
        clear_activations_per_step=True,
    )

    hook_manager = HookManager(hook_config)

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
        return False
    finally:
        if hasattr(transformer, "remove_hooks"):
            transformer.remove_hooks()
        if hasattr(transformer, "clear_activations"):
            transformer.clear_activations()

    print(f"  Capture complete. Temporary files in {run_temp_dir}")
    return True


# Main driver

def main() -> None:
    print("\n" + "=" * 80)
    print("DiT ACTIVATION EXTRACTION → WEBDATASET (PER SPEAKER)")
    print("=" * 80 + "\n")

    LIBRITTS_ROOT = Path("/home/ewong/links/scratch/LibriTTS")
    SUBSET_TO_PROCESS = "test-clean"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    OUTPUT_DIR = Path("./extracted_features_sharded")
    TRAIN_TARS_DIR = OUTPUT_DIR / "train_tars"
    TEST_TARS_DIR = OUTPUT_DIR / "test_tars"
    TEMP_DIR = OUTPUT_DIR / "tmp"

    TRAIN_TARS_DIR.mkdir(parents=True, exist_ok=True)
    TEST_TARS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        model, vocoder = load_f5_model_and_vocoder(device=DEVICE)
    except Exception as e:
        print(f"Fatal error loading model/vocoder: {e}")
        return

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

    subset_dir = LIBRITTS_ROOT / SUBSET_TO_PROCESS
    if not subset_dir.exists():
        print(f"ERROR: Subset directory {subset_dir} does not exist.")
        return

    speaker_dirs = sorted([p for p in subset_dir.iterdir() if p.is_dir()])
    print(f"Found {len(speaker_dirs)} speakers in subset '{SUBSET_TO_PROCESS}'.")

    successful_count = 0
    failed_count = 0

    for idx, speaker_dir in enumerate(speaker_dirs):
        # Limit for testing. Remove to run all speakers.
        if idx == 2: 
            break
            
        speaker_id = speaker_dir.name
        print(
            f"\n{'=' * 80}\n"
            f"Processing Speaker {idx + 1}/{len(speaker_dirs)}: {speaker_id}\n"
            f"{'=' * 80}"
        )

        wav_files = sorted(list(speaker_dir.rglob("*.wav")))
        if len(wav_files) < 2:
            print(f"  Skipping speaker {speaker_id}: need at least 2 WAV files.")
            failed_count += 1
            continue

        train_wav = wav_files[0]
        test_wav = wav_files[1]
        train_txt = train_wav.with_suffix(".normalized.txt")
        test_txt = test_wav.with_suffix(".normalized.txt")

        if not (train_txt.exists() and test_txt.exists()):
            print(f"  Skipping speaker {speaker_id}: missing normalized text.")
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

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Successfully processed: {successful_count}")
    print(f"Failed: {failed_count}")

    if TEMP_DIR.exists():
        try:
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
        except Exception as e:
            print(f"Warning: could not fully clean TEMP_DIR: {e}")


if __name__ == "__main__":
    main()