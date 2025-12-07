"""
Step 2: DiT Activation Extraction (Train/Test/Eval Split)
"""
from __future__ import annotations
from pathlib import Path
import json
import shutil
import soundfile as sf
from typing import Dict, List, Tuple
import torch
from datasets import load_from_disk
from omegaconf import OmegaConf
from hydra.utils import get_class
from importlib.resources import files
from f5_tts.infer.utils_infer import load_model, load_vocoder, preprocess_ref_audio_text, infer_process
from f5_tts.model.cfm import HookConfig, HookManager

# --- CONFIGURATION ---
INPUT_DATASET_PATH = "./libritts_reference_speakers"
OUTPUT_DIR = Path("./extracted_features_fixed_text")

TRAIN_TEXT = "The rainbow is a division of white light into many beautiful colors."
TEST_TEXT = "The quick brown fox jumps over the lazy dog."
EVAL_TEXT = "The quick brown fox jumps over the lazy dog." 

# Directories
DIRS = {
    "train": { "tensors": OUTPUT_DIR / "train" / "tensors", "index": OUTPUT_DIR / "train_index.json" },
    "test":  { "tensors": OUTPUT_DIR / "test" / "tensors",  "index": OUTPUT_DIR / "test_index.json" },
    "eval":  { "tensors": OUTPUT_DIR / "eval" / "tensors",  "index": OUTPUT_DIR / "eval_index.json" }
}
TEMP_DIR = OUTPUT_DIR / "tmp"

for split in DIRS.keys():
    DIRS[split]["tensors"].mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# 1. Model Loading
def load_f5_model_and_vocoder(device: str = "cuda") -> Tuple[torch.nn.Module, torch.nn.Module]:
    print(f"Device: {device}")
    model_name = "F5TTS_v1_Base"
    LOCAL_MODEL_PATH = Path("/home/ewong/links/scratch/checkpoints/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors")
    LOCAL_VOCODER_PATH = Path("/home/ewong/links/scratch/checkpoints/vocos-mel-24khz")

    if not LOCAL_MODEL_PATH.exists(): raise FileNotFoundError(f"Missing model at {LOCAL_MODEL_PATH}")

    print("Loading vocoder...")
    vocoder = load_vocoder(vocoder_name="vocos", is_local=True, local_path=str(LOCAL_VOCODER_PATH), device=device)
    
    print("Loading F5-TTS model...")
    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model_name}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model = load_model(
        model_cls, model_cfg.model.arch, str(LOCAL_MODEL_PATH),
        mel_spec_type="vocos", vocab_file="", device=device
    )
    print("Model loaded\n")
    return model, vocoder

# 2. Layer Selection
def select_dit_hook_layers(transformer: torch.nn.Module) -> Dict[str, str]:
    hook_layers = {}
    for name, _ in transformer.named_modules():
        parts = name.split(".")
        if len(parts) == 3 and parts[0] == "transformer_blocks" and parts[2] in ["attn_norm", "attn", "ff"]:
            hook_layers[name] = name
    return hook_layers

# 3. Capture Logic
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
    
    base_tensor_dir = DIRS[split]["tensors"]
    speaker_tensor_dir = base_tensor_dir / speaker_id
    speaker_tensor_dir.mkdir(parents=True, exist_ok=True)
    
    run_temp_dir = TEMP_DIR / f"{speaker_id}_{split}"
    run_temp_dir.mkdir(parents=True, exist_ok=True)

    transformer = model.transformer
    capture_layers = select_dit_hook_layers(transformer)

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
        shutil.rmtree(run_temp_dir, ignore_errors=True)
        return []
    finally:
        if hasattr(transformer, "remove_hooks"): transformer.remove_hooks()
        if hasattr(transformer, "clear_activations"): transformer.clear_activations()

    entries = []
    
    for step_folder in sorted(run_temp_dir.iterdir()):
        if not step_folder.is_dir(): continue
        try:
            if step_folder.name.startswith("step_"):
                step_idx = int(step_folder.name.split("_")[1])
            else:
                step_idx = int(step_folder.name)
        except ValueError:
            continue
            
        step_tensors = {}
        for pt_file in step_folder.glob("*.pt"):
            try:
                data = torch.load(pt_file, map_location="cpu")
                layer_name = pt_file.stem
                if "activations" in data and "__output__" in data["activations"]:
                    step_tensors[layer_name] = data["activations"]["__output__"]
            except Exception:
                continue

        if step_tensors:
            combined_filename = f"step_{step_idx}.pt"
            combined_path = speaker_tensor_dir / combined_filename
            torch.save(step_tensors, combined_path)
            
            entries.append({
                "speaker_id": speaker_id,
                "split": split,
                "gen_text": target_gen_text,
                "pt_file": str(combined_path.absolute()),
                "timestep": step_idx,
                "layers_included": list(step_tensors.keys())
            })

    shutil.rmtree(run_temp_dir, ignore_errors=True)
    return entries

# 4. Main Driver
def main() -> None:
    print("\n" + "=" * 80)
    print("DiT EXTRACTION (Train/Test/Eval)")
    print("=" * 80 + "\n")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model, vocoder = load_f5_model_and_vocoder(device=DEVICE)
        print(f"Loading reference dataset from {INPUT_DATASET_PATH}...")
        ds = load_from_disk(INPUT_DATASET_PATH)
        print(f"Dataset loaded. Total samples: {len(ds)}")
    except Exception as e:
        print(f"Fatal error: {e}")
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

    indexes = { "train": [], "test": [], "eval": [] }
    
    for idx, sample in enumerate(ds):
        speaker_id = str(sample["speaker_id"])
        sample_idx = sample.get("sample_idx", 0)
        
        print(f"\nProcessing {idx+1}/{len(ds)}: Speaker {speaker_id} | Sample {sample_idx}")

        # Write current sample to wav
        temp_wav_path = TEMP_DIR / f"{speaker_id}_sample{sample_idx}.wav"
        sf.write(str(temp_wav_path), sample["audio"]["array"], sample["audio"]["sampling_rate"])

        if sample_idx == 0:
            # Used for both Train and Test generations
            print(f"  > Generating TRAIN...")
            train_entries = run_speaker_capture(
                model, vocoder, speaker_id, temp_wav_path,
                sample["text_normalized"], TRAIN_TEXT,
                inference_params, DEVICE, "train"
            )
            indexes["train"].extend(train_entries)

            print(f"  > Generating TEST...")
            test_entries = run_speaker_capture(
                model, vocoder, speaker_id, temp_wav_path,
                sample["text_normalized"], TEST_TEXT,
                inference_params, DEVICE, "test"
            )
            indexes["test"].extend(test_entries)
        
        elif sample_idx == 1:
            # Used for Eval generation (checks generalization to new audio)
            print(f"  > Generating EVAL...")
            eval_entries = run_speaker_capture(
                model, vocoder, speaker_id, temp_wav_path,
                sample["text_normalized"], EVAL_TEXT,
                inference_params, DEVICE, "eval"
            )
            indexes["eval"].extend(eval_entries)

        if temp_wav_path.exists(): temp_wav_path.unlink()

        # Incremental Save
        if (idx + 1) % 10 == 0:
            for split in indexes:
                with open(DIRS[split]["index"], "w") as f: json.dump(indexes[split], f, indent=2)

    # Final Save
    for split in indexes:
        with open(DIRS[split]["index"], "w") as f: json.dump(indexes[split], f, indent=2)
    
    if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    main()