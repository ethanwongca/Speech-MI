from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, Dict

import torch
from omegaconf import OmegaConf
from hydra.utils import get_class
from importlib.resources import files

# F5-TTS imports
# We only need loading utilities, not the inference process or HookManager
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
)

# -------------------------------------------------------------------------
# 1. Model Loading (Same configuration as your pipeline)
# -------------------------------------------------------------------------
def load_f5_model_and_vocoder(device: str = "cuda") -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Load the F5-TTS model and vocoder using local disk paths."""
    print(f"Device: {device}")

    model_name = "F5TTS_v1_Base"
    
    # --- USER PATHS (Keep these matching your environment) ---
    LOCAL_MODEL_PATH = Path(
        "/home/ewong/links/scratch/checkpoints/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    )
    LOCAL_VOCODER_PATH = Path(
        "/home/ewong/links/scratch/checkpoints/vocos-mel-24khz"
    )
    # ---------------------------------------------------------

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
    # Load config from the installed f5_tts package
    model_cfg = OmegaConf.load(
        str(files("f5_tts").joinpath(f"configs/{model_name}.yaml"))
    )
    
    # Dynamically get the class (e.g. DiT) based on config
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

# -------------------------------------------------------------------------
# 2. Hierarchy Printer (Replaces layer selection)
# -------------------------------------------------------------------------
def print_dit_hierarchy(transformer: torch.nn.Module) -> None:
    """
    Prints all layers with hierarchical depth to help verify structure.
    """
    print(f"\n{'='*20} DiT Model Hierarchy {'='*20}")

    for name, module in transformer.named_modules():
        # Skip the root module
        if name == "":
            continue

        # Calculate visual indentation based on dot depth
        depth = name.count(".")
        indent = "    " * depth
        
        # Print: Indent |-- LayerName (ClassType)
        print(f"{indent}|-- {name} ({module.__class__.__name__})")

    print(f"{'='*63}")
    print("Hierarchy printing complete.")


# -------------------------------------------------------------------------
# 3. Main Execution
# -------------------------------------------------------------------------
def main() -> None:
    print("\n" + "=" * 80)
    print("F5-TTS MODEL HIERARCHY INSPECTOR")
    print("=" * 80 + "\n")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Load the model exactly as the inference script would
        model, _ = load_f5_model_and_vocoder(device=DEVICE)
        
        # Access the DiT transformer
        if hasattr(model, "transformer"):
            transformer = model.transformer
        else:
            print("Error: 'transformer' attribute not found on model.")
            return

        # Print the structure
        print_dit_hierarchy(transformer)

    except Exception as e:
        print(f"Fatal error: {e}")
        return

if __name__ == "__main__":
    main()