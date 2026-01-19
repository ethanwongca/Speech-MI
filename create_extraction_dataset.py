"""
Step 2: DiT Activation Extraction (Mean Vector Split)
"""
from __future__ import annotations
from pathlib import Path
import json
import shutil
import re
import time
import soundfile as sf
from typing import Dict, List, Tuple, Set
import torch
from datasets import load_from_disk
from omegaconf import OmegaConf
from hydra.utils import get_class
from importlib.resources import files
from f5_tts.infer.utils_infer import load_model, load_vocoder, preprocess_ref_audio_text, infer_process
from f5_tts.model.cfm import HookConfig, HookManager

# --- CONFIGURATIONS ---
INPUT_DATASET_PATH = "./libritts_reference_speakers"
OUTPUT_DIR = Path("./mean_vectors")

TRAIN_TEXT = "The rainbow is a division of white light into many beautiful colors."
TEST_TEXT = "The quick brown fox jumps over the lazy dog."
EVAL_TEXT = "The quick brown fox jumps over the lazy dog." 

DIRS = {
    "train": { "folder": OUTPUT_DIR / "train", "index": OUTPUT_DIR / "train_index.jsonl" },
    "test":  { "folder": OUTPUT_DIR / "test",  "index": OUTPUT_DIR / "test_index.jsonl" },
    "eval":  { "folder": OUTPUT_DIR / "eval",  "index": OUTPUT_DIR / "eval_index.jsonl" }
}
TEMP_DIR = OUTPUT_DIR / "tmp"

# Utility functions

def mean_vector(layer):
    """
    Converts attn layer and FFN layer to a mean vector
    Arguments: 
        layer: torch.tensor, that is three dimensions [B, T, D_model]
    Output:
        mean_vector: torch.tensor with two dimensions [D_model]
    """
    # from dit.py extract the conditional path of DiT 
    # it's index 0 due to x = torch.cat((x_cond, x_uncond), dim=0) 
    # cond, and uncond make up the batch dimensions
    cond_layer = layer[0, :, :]
    # cond [T, D_model]
    mv = torch.mean(cond_layer, 0)
    # [D_model]
    return mv

def get_existing_ids(index_path: Path) -> Set[str]:
    existing = set()
    if index_path.exists():
        with open(index_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing.add(str(data["speaker_id"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return existing

def load_f5_model_and_vocoder(device: str = "cuda") -> Tuple[torch.nn.Module, torch.nn.Module]:
    model_name = "F5TTS_v1_Base"
    LOCAL_MODEL_PATH = Path("/home/ewong/links/scratch/checkpoints/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors")
    LOCAL_VOCODER_PATH = Path("/home/ewong/links/scratch/checkpoints/vocos-mel-24khz")

    if not LOCAL_MODEL_PATH.exists(): raise FileNotFoundError(f"Missing model at {LOCAL_MODEL_PATH}")
    
    vocoder = load_vocoder(vocoder_name="vocos", is_local=True, local_path=str(LOCAL_VOCODER_PATH), device=device)
    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model_name}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model = load_model(
        model_cls, model_cfg.model.arch, str(LOCAL_MODEL_PATH),
        mel_spec_type="vocos", vocab_file="", device=device
    )
    return model, vocoder

def select_dit_hook_layers(transformer: torch.nn.Module) -> Dict[str, str]:
    hook_layers = {}
    for name, _ in transformer.named_modules():
        parts = name.split(".")
        # Selection now includes both 'attn' and 'ff'
        if len(parts) == 3 and parts[0] == "transformer_blocks" and parts[2] in ["attn", "ff"]:
            hook_layers[name] = name
    return hook_layers

def run_speaker_capture(
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    speaker_id: str,
    wav_path: Path,
    ref_text_prompt: str,
    target_gen_text: str,
    inference_params: dict,
    device: str,
    split: str,
) -> List[Dict]:
    
    # Ensure permanent directory exists
    speaker_dir = DIRS[split]["folder"] / speaker_id
    speaker_dir.mkdir(parents=True, exist_ok=True)
    
    run_temp_dir = TEMP_DIR / f"{speaker_id}_{split}"
    if run_temp_dir.exists():
        shutil.rmtree(run_temp_dir)
    run_temp_dir.mkdir(parents=True, exist_ok=True)

    transformer = model.transformer
    capture_layers = select_dit_hook_layers(transformer)

    hook_config = HookConfig(
        capture_layers=capture_layers,
        precision=torch.bfloat16,
        device=device,
        save_per_timestep=True,
        save_dir=str(run_temp_dir),
        clear_activations_per_step=True,
    )
    hook_manager = HookManager(hook_config)

    try:
        ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(str(wav_path), ref_text_prompt)
        _ = infer_process(
            ref_audio=ref_audio_processed,
            ref_text=ref_text_processed,
            gen_text=target_gen_text,
            model_obj=model,
            vocoder=vocoder,
            hook_manager=hook_manager,
            **inference_params,
        )
    except Exception as e:
        print(f"  Capture failed for {speaker_id} ({split}): {e}")
        return []
    finally:
        if hasattr(transformer, "remove_hooks"): transformer.remove_hooks()
        if hasattr(transformer, "clear_activations"): transformer.clear_activations()

    # --- UPDATED CONSOLIDATION LOGIC ---
    print(f"  > Consolidating vectors for {speaker_id}...")
    entries = []
    
    # Use rglob to find all .pt files inside the step_XXXX folders
    all_pt_files = list(run_temp_dir.rglob("*.pt"))
    
    # Group files by their timestep index (e.g., extracting 0 from 'step_0000')
    timestep_groups = {}
    for pt_file in all_pt_files:
        # Match 'step_XXXX' in the path
        match = re.search(r'step_(\d+)', str(pt_file))
        if match:
            step_idx = int(match.group(1))
            if step_idx not in timestep_groups:
                timestep_groups[step_idx] = []
            timestep_groups[step_idx].append(pt_file)

    for step_idx in sorted(timestep_groups.keys()):
        attn_tensors, ff_tensors = {}, {}

        for pt_file in timestep_groups[step_idx]:
            try:
                data = torch.load(pt_file, map_location="cpu")
                layer_name = pt_file.stem  # e.g., transformer_blocks_0_attn
                
                # Extract tensor from the saved hook format
                raw_tensor = None
                if isinstance(data, torch.Tensor):
                    raw_tensor = data
                elif isinstance(data, dict):
                    if "activations" in data and "__output__" in data["activations"]:
                        raw_tensor = data["activations"]["__output__"]
                    elif "tensor" in data:
                        raw_tensor = data["tensor"]

                if raw_tensor is not None:
                    # Apply your mean_vector logic: [2, T, 1024] -> [1024]
                    pooled = mean_vector(raw_tensor)
                    
                    # Sort into Attn or FF groups based on filename from screenshot
                    if "_attn" in layer_name:
                        attn_tensors[layer_name] = pooled
                    elif "_ff" in layer_name:
                        ff_tensors[layer_name] = pooled
                
                del data 
            except Exception:
                continue

        # Save the decoupled containers for the step
        for group_name, tensors in [("attn", attn_tensors), ("ff", ff_tensors)]:
            if not tensors: continue
            
            # This saves the files exactly as you requested
            filename = f"step_{step_idx}_{group_name}.pt"
            dest_path = speaker_dir / filename
            
            torch.save(tensors, dest_path)
            
            entries.append({
                "speaker_id": speaker_id,
                "split": split,
                "type": group_name,
                "pt_file": str(dest_path.absolute()),
                "timestep": step_idx,
                "layers_included": list(tensors.keys())
            })

    print(f"  > Successfully generated {len(entries)} permanent files.")
    shutil.rmtree(run_temp_dir, ignore_errors=True)
    return entries

# appending to jsonl

def append_to_jsonl(entries: List[Dict], filepath: Path):
    if not entries: return
    with open(filepath, "a") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

def main() -> None:
    print("\n" + "=" * 80)
    print("DiT DECOUPLED MEAN VECTOR EXTRACTION")
    print("=" * 80 + "\n")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    for split in DIRS.keys():
        DIRS[split]["folder"].mkdir(parents=True, exist_ok=True)
        if not DIRS[split]["index"].exists():
            with open(DIRS[split]["index"], "w") as f: pass
            
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    done_train = get_existing_ids(DIRS["train"]["index"])
    done_test = get_existing_ids(DIRS["test"]["index"])
    done_eval = get_existing_ids(DIRS["eval"]["index"])

    try:
        model, vocoder = load_f5_model_and_vocoder(device=DEVICE)
        ds = load_from_disk(INPUT_DATASET_PATH)
    except Exception as e:
        print(f"Fatal error: {e}"); return

    inference_params = {
        "mel_spec_type": "vocos", "target_rms": 0.1, "cross_fade_duration": 0.15, "nfe_step": 32,
        "cfg_strength": 2.0, "sway_sampling_coef": -1, "speed": 1.0, "fix_duration": None, "device": DEVICE,
    }

    for idx, sample in enumerate(ds):
        speaker_id = str(sample["speaker_id"])
        sample_idx = sample.get("sample_idx", 0)

        # continuation logic
        if sample_idx == 0:
            if speaker_id in done_train and speaker_id in done_test: continue
        elif sample_idx == 1:
            if speaker_id in done_eval: continue
        
        print(f"\nProcessing {idx+1}/{len(ds)}: Speaker {speaker_id} | Sample {sample_idx}")
        temp_wav_path = TEMP_DIR / f"{speaker_id}_sample{sample_idx}.wav"
        sf.write(str(temp_wav_path), sample["audio"]["array"], sample["audio"]["sampling_rate"])

        if sample_idx == 0:
            if speaker_id not in done_train:
                print(f"  > TRAIN...")
                append_to_jsonl(run_speaker_capture(model, vocoder, speaker_id, temp_wav_path, sample["text_normalized"], TRAIN_TEXT, inference_params, DEVICE, "train"), DIRS["train"]["index"])
            if speaker_id not in done_test:
                print(f"  > TEST...")
                append_to_jsonl(run_speaker_capture(model, vocoder, speaker_id, temp_wav_path, sample["text_normalized"], TEST_TEXT, inference_params, DEVICE, "test"), DIRS["test"]["index"])
        elif sample_idx == 1:
            if speaker_id not in done_eval:
                print(f"  > EVAL...")
                append_to_jsonl(run_speaker_capture(model, vocoder, speaker_id, temp_wav_path, sample["text_normalized"], EVAL_TEXT, inference_params, DEVICE, "eval"), DIRS["eval"]["index"])

        if temp_wav_path.exists(): temp_wav_path.unlink()

    if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    main()