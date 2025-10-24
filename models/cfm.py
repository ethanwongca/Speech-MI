"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

print("\n\n<<<<<<<<<< LOADING MODIFIED CFM WITH HOOKS >>>>>>>>>>\n\n")

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)

class HookConfig:
    """
    Configuration object for managing hooks during ODE sampling.
    Uses DiT's existing hook infrastructure (add_extraction_hook, add_patch_hook).
    
    Three simple capture modes:
    1. capture_all_layers=True: Captures ALL transformer blocks
    2. capture_layers: Dict of specific layers {hook_name: layer_path}
    3. capture_final_output: Captures the final output layer
    
    Time filtering controls WHEN to capture (which timesteps).
    """
    def __init__(
        self,
        capture_all_layers: bool = False,
        # Input for this is {hook_name: layer_path}
        capture_layers: dict[str, str] | None = {}, 
        capture_final_output: bool = False, 
        # Invoke a  Lambda function in time filter i.e. time_filter=lambda step for step in [0, 10, 20, 31] etc 
        time_filter: Callable[[float, int], bool] | None = None,  
        precision: torch.dtype = torch.bfloat16,
        device: str = 'cpu',
        save_per_timestep: bool = True,
        save_dir: str | Path = "./hook_captures",
        patches: dict | None = {}
    ): 
        self.capture_all_layers = capture_all_layers
        self.capture_layers = capture_layers
        self.capture_final_output = capture_final_output
        # Set default vlaues to capturea all time steps 
        self.time_filter = time_filter or (lambda t, step: True)
        self.precision = precision
        self.device = device
        self.save_per_timestep = save_per_timestep
        self.save_dir = Path(save_dir)
        if self.save_per_timestep:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.patches = patches
        self._saved_files = []
        self._metadata = {
            'precision': str(precision),
            'device': device,
            'capture_all_layers': capture_all_layers,
            'capture_layers': list(self.capture_layers.keys()),
            'timesteps': []
        }
    def should_capture_time(self, t: float, step_index: int) -> bool:
        """Check if this timestep should be captured"""
        return self.time_filter(t, step_index)
    
    def get_layers_to_hook(self, transformer_module: nn.Module) -> dict[str, str]:
        """
        Returns a dict of {hook_name: layer_path} for all layers to capture.
        """
        hooks_dict = {}
        
        # 1. Capture all transformer blocks if requested
        if self.capture_all_layers:
            if hasattr(transformer_module, 'transformer_blocks'):
                for i, _ in enumerate(transformer_module.transformer_blocks):
                    hooks_dict[f'block_{i}'] = f'transformer_blocks.{i}'
        
        # 2. Add manually specified layers
        hooks_dict.update(self.capture_layers)
        
        # 3. Add final output if requested
        if self.capture_final_output:
            hooks_dict['final_output'] = 'proj_out'
        
        return hooks_dict
    
    def save_timestep(self, t: float, step_index: int, activations: dict[str, torch.Tensor]) -> str:
        """
        Saves activations for a single timestep to disk.
        Returns the filepath where data was saved.
        """
        if not self.save_per_timestep:
            return None
        
        timestep_data = {}
        for hook_name, tensor in activations.items():
            timestep_data[hook_name] = {
                'tensor': tensor.to(self.precision).to('cpu'),
                'shape': tuple(tensor.shape),
                'dtype': str(tensor.dtype),
                'original_device': str(tensor.device)
            }
        
        # Create filename with timestep counter and t value
        filename = f"timestep_{step_index:04d}_t_{t:.6f}.pt"
        filepath = self.save_dir / filename
        
        # Save to disk
        torch.save({
            't': t,
            'step_index': step_index,
            'activations': timestep_data
        }, filepath)
        
        # Update metadata
        self._metadata['timesteps'].append({
            'step_index': step_index,
            't': t,
            'file': filename,
            'num_hooks': len(timestep_data)
        })
        
        self._saved_files.append(str(filepath))
        
        print(f"Saved timestep {step_index} (t={t:.6f}) with {len(timestep_data)} hooks to {filename}")
        
        return str(filepath)
    
    def save_metadata(self):
        """Save metadata about the capture session"""
        if not self.save_per_timestep:
            return
        
        metadata_path = self.save_dir / "capture_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self._metadata, f, indent=2)
        
        print(f"Saved capture metadata to {metadata_path}")
    
    def get_saved_files(self) -> list[str]:
        """Returns list of all saved timestep files"""
        return self._saved_files
    
    def load_timestep(self, timestep_index: int) -> dict:
        """Load a specific timestep from disk"""
        if timestep_index >= len(self._saved_files):
            raise IndexError(f"Timestep {timestep_index} not found. Only {len(self._saved_files)} timesteps saved.")
        
        filepath = self._saved_files[timestep_index]
        return torch.load(filepath)
    
    @staticmethod
    def load_from_dir(save_dir: str | Path) -> list[dict]:
        """
        Load all timesteps from a directory.
        Returns list of timestep data dicts.
        """
        save_dir = Path(save_dir)
        timestep_files = sorted(save_dir.glob("timestep_*.pt"))
        
        all_timesteps = []
        for filepath in timestep_files:
            all_timesteps.append(torch.load(filepath))
        
        return all_timesteps

class HookManager:
    """
    Manages the CFM.sample() and uses the Hook Classes defined in DiT 

    i.e. DiT contains 
    transformer.add_extraction_hook(submodule_path, hook_name)
    transformer.add_patch_hook(submodule_path, hook_name, **patch_kwargs)
    transformer.remove_hooks()
    transformer.clear_activations()
    transformer.extracted_activations (dict)
    """

    def __init__(self, config: HookConfig):
        self.config = config
        self._hooks_attached = False
        self._hook_names = []
        self._step_counter = 0
    
    def setup_hooks(self, transformer_module: nn.Module):
        """
        Attach all hooks specified in config using DiT's native hook methods.
        """
        # Get all layers to hook
        layers_to_hook = self.config.get_layers_to_hook(transformer_module)
        
        if not layers_to_hook:
            print("No layers to hook!")
            return
        
        print(f"\n{'='*70}")
        print(f"HOOK SETUP")
        print(f"{'='*70}")
        print(f"Attaching {len(layers_to_hook)} extraction hooks:")
        
        # Attach extraction hooks using DiT's method
        for hook_name, layer_path in layers_to_hook.items():
            try:
                transformer_module.add_extraction_hook(
                    submodule_path=layer_path,
                    hook_name=hook_name
                )
                self._hook_names.append(hook_name)
                print(f"{hook_name:30s} → {layer_path}")
            except Exception as e:
                print(f"Error: {hook_name:30s} → {layer_path} (ERROR: {e})")
        
        # Attach patch hooks if specified using DiT's method
        if self.config.patches:
            print(f"\n🔧 Attaching {len(self.config.patches)} patch hooks:")
            for hook_name, patch_kwargs in self.config.patches.items():
                layer_path = layers_to_hook.get(hook_name)
                if layer_path:
                    try:
                        transformer_module.add_patch_hook(
                            submodule_path=layer_path,
                            hook_name=f"{hook_name}_patch",
                            **patch_kwargs
                        )
                        print(f"{hook_name}_patch → {layer_path}")
                    except Exception as e:
                        print(f"{hook_name}_patch → {layer_path} (ERROR: {e})")
        
        print(f"{'='*70}\n")
        self._hooks_attached = True
    
    def capture_and_save_timestep(self, transformer_module: nn.Module, t: float) -> bool:
        """
        Capture activations for current timestep from DiT's extracted_activations,
        save to disk, then clear.
        
        Returns True if captured, False if skipped.
        """
        if not self._hooks_attached:
            return False
        
        step_index = self._step_counter
        self._step_counter += 1
        
        # Check if we should capture this timestep
        if not self.config.should_capture_time(t, step_index):
            transformer_module.clear_activations()
            return False
        
        # Collect all activations from DiT's extracted_activations dict
        activations = {}
        for hook_name in self._hook_names:
            if hook_name in transformer_module.extracted_activations:
                captured_data = transformer_module.extracted_activations[hook_name]
                # DiT stores as {'tensor': ..., 'device': ..., 'shape': ..., 'call_count': ...}
                activations[hook_name] = captured_data['tensor']
        
        # Save to disk if we captured anything
        if activations:
            self.config.save_timestep(t, step_index, activations)
        
        # Clear activations to free memory (uses DiT's method)
        transformer_module.clear_activations()
        
        return True
    
    def cleanup(self, transformer_module: nn.Module):
        """
        Remove all hooks and save metadata using DiT's cleanup methods.
        """
        if self._hooks_attached:
            # Use DiT's native cleanup methods
            transformer_module.remove_hooks()
            transformer_module.clear_activations()
            
            # Save our metadata
            self.config.save_metadata()
            
            print(f"\n{'='*70}")
            print(f"Capture complete!")
            print(f"Total ODE steps: {self._step_counter}")
            print(f"Timesteps saved: {len(self.config.get_saved_files())}")
            print(f"Save directory: {self.config.save_dir}")
            print(f"{'='*70}\n")
            
            self._hooks_attached = False



class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        duration: int | int["b"],
        *,
        lens: int["b"] | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
        hook_manager: HookManager | None = None,
    ):
        self.eval()

        if hook_manager:
            hook_manager.setup_hooks(self.transformer)

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

        if batch > 1:
            mask = lens_to_mask(duration)
        else:
            mask = None

        # neural ode
        def fn(t, x):
            # predict flow
            if cfg_strength < 1e-5:
                pred = self.transformer(x=x, cond=step_cond, text=text, time=t, mask=mask, drop_audio_cond=False, drop_text=False, cache=True)
            else:
                pred_cfg = self.transformer(x=x, cond=step_cond, text=text, time=t, mask=mask, cfg_infer=True, cache=True)
                pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
                pred = pred + (pred - null_pred) * cfg_strength
            
            # If hooks were requested, capture their data for this specific timestep.
            if hook_manager:
                t_float = t.item() if t.ndim == 0 else t[0].item()
                hook_manager.capture_and_save_timestep(self.transformer, t_float)

            return pred

        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss:
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        # If we attached hooks for this run, make sure to clean them up.
        if hook_manager:
            hook_manager.cleanup(self.transformer)

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        *,
        lens: int["b"] | None = None,
        noise_scheduler: str | None = None,
    ):
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)

        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        x1 = inp
        x0 = torch.randn_like(x1)

        time = torch.rand((batch,), dtype=dtype, device=self.device)

        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        drop_audio_cond = random() < self.audio_drop_prob
        if random() < self.cond_drop_prob:
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        pred = self.transformer(
            x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask
        )

        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred