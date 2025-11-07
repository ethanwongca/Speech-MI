"""
Fixed OLD CFM system with proper memory management
"""

from __future__ import annotations

print("\n\n<<<<<<<<<< LOADING FIXED CFM WITH PROPER MEMORY MANAGEMENT >>>>>>>>>>\n\n")

from random import random
from typing import Callable
import gc

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
from pathlib import Path
import json
from collections import defaultdict

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

def _get_group_and_key(hook_name: str) -> tuple[str, str]:
    """Smart grouping for saved activations."""
    parts = hook_name.split('.')
    if parts[0] == 'transformer_blocks' and len(parts) > 1 and parts[1].isdigit():
        group_name = f'transformer_blocks.{parts[1]}'
        sub_key = '.'.join(parts[2:]) if len(parts) > 2 else '__output__'
    elif '.' in hook_name:
        group_name = parts[0]
        sub_key = '.'.join(parts[1:])
    else:
        group_name = hook_name
        sub_key = '__output__'
    return group_name, sub_key


class HookConfig:
    """Configuration for hook management during ODE sampling."""
    
    def __init__(
        self,
        capture_all_submodules: bool = False,
        capture_all_layers: bool = False,
        capture_layers: dict[str, str] | None = None,
        capture_final_output: bool = False,
        time_filter: Callable[[float, int], bool] | None = None,
        precision: torch.dtype = torch.bfloat16,
        device: str = 'cpu',
        save_per_timestep: bool = True,
        save_dir: str | Path = "./hook_captures",
        patches: dict[int, list[dict]] | None = None,
        clear_activations_per_step: bool = True
    ):
        self.capture_all_submodules = capture_all_submodules
        self.capture_all_layers = capture_all_layers
        self.capture_layers = capture_layers or {}
        self.capture_final_output = capture_final_output
        self.time_filter = time_filter or (lambda t, step: True)
        self.precision = precision
        self.device = device
        self.save_per_timestep = save_per_timestep
        self.save_dir = Path(save_dir)
        if self.save_per_timestep:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.patches = patches or {}
        self.clear_activations_per_step = clear_activations_per_step
        
        self._saved_files = []
        self._metadata = {
            'precision': str(precision),
            'device': device,
            'capture_all_submodules': capture_all_submodules,
            'capture_all_layers': capture_all_layers,
            'capture_layers': list(self.capture_layers.keys()),
            'timesteps': [],
        }
    
    def should_capture_time(self, t: float, step_index: int) -> bool:
        return self.time_filter(t, step_index)
    
    def get_layers_to_hook(self, transformer_module: nn.Module) -> dict[str, str]:
        """Returns dict of {hook_name: layer_path} for all layers to capture."""
        hooks_dict = {}
        
        if self.capture_all_submodules:
            print(f"[HookConfig] Capturing ALL submodules...")
            for module_path, module in transformer_module.named_modules():
                if not module_path:  # Skip root
                    continue
                hooks_dict[module_path] = module_path
        
        if self.capture_all_layers:
            if hasattr(transformer_module, 'transformer_blocks'):
                for i in range(len(transformer_module.transformer_blocks)):
                    hooks_dict[f'block_{i}'] = f'transformer_blocks.{i}'
        
        hooks_dict.update(self.capture_layers)
        
        if self.capture_final_output:
            hooks_dict['final_output'] = 'proj_out'
        
        num_hooks = len(hooks_dict)
        if num_hooks > 0:
            print(f"[HookConfig] Will attach {num_hooks} total hooks")
        
        return hooks_dict
    
    def save_timestep(
        self,
        t: float,
        step_index: int,
        activations: dict[str, torch.Tensor]
    ) -> list[str]:
        """
        Save activations with PROPER memory management.
        KEY FIX: Explicitly detach, clone, move to CPU, and delete references.
        """
        if not self.save_per_timestep:
            return []
        
        saved_files = []
        timestep_dir = self.save_dir / f"step_{step_index:04d}"
        timestep_dir.mkdir(parents=True, exist_ok=True)
        
        # Group activations
        grouped_tensors = defaultdict(dict)
        
        for hook_name, tensor in activations.items():
            group_name, sub_key = _get_group_and_key(hook_name)
            
            # ✅ CRITICAL FIX: Proper tensor copying
            # 1. Detach from computation graph
            # 2. Clone to create new memory
            # 3. Convert precision
            # 4. Move to CPU (creates another copy)
            cpu_tensor = tensor.detach().clone().to(dtype=self.precision, device='cpu', non_blocking=False)
            
            grouped_tensors[group_name][sub_key] = cpu_tensor
            
            # ✅ Explicitly delete the CPU copy reference
            del cpu_tensor
        
        # Save each group
        num_groups = len(grouped_tensors)
        for group_name, tensor_dict in grouped_tensors.items():
            safe_filename = f"{group_name.replace('/', '_').replace('.', '_')}.pt"
            filepath = timestep_dir / safe_filename
            
            torch.save({
                't': t,
                'step_index': step_index,
                'group_name': group_name,
                'activations': tensor_dict,
            }, filepath)
            
            saved_files.append(str(filepath))
        
        # ✅ Delete the grouped dict
        del grouped_tensors
        
        # Update metadata
        self._metadata['timesteps'].append({
            'step_index': step_index,
            't': t,
            'directory': str(timestep_dir),
            'num_groups': num_groups,
            'group_names': list(activations.keys()),
        })
        
        self._saved_files.extend(saved_files)
        
        # ✅ Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Saved step {step_index} (t={t:.6f}): {len(activations)} hooks → {num_groups} groups | GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        return saved_files
    
    def save_metadata(self):
        if not self.save_per_timestep:
            return
        metadata_path = self.save_dir / "capture_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self._metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")
    
    def get_saved_files(self) -> list[str]:
        return self._saved_files


class HookManager:
    """Manages hooks during CFM sampling with proper extraction support."""

    def __init__(self, config: HookConfig):
        self.config = config
        self._hooks_attached = False
        self._hook_names = []
        self._step_counter = 0
        self._active_patch_hooks: list[str] = []
    
    def setup_hooks(self, transformer_module: nn.Module):
        """Attach all hooks specified in config using DiT's native hook methods."""
        transformer_module.clear_activations()
        layers_to_hook = self.config.get_layers_to_hook(transformer_module)
        
        if not layers_to_hook and not self.config.patches:
            print("No layers to hook!")
            return
        
        print(f"\n{'='*70}")
        print(f"HOOK SETUP")
        print(f"{'='*70}")
        
        # Attach extraction hooks
        if layers_to_hook:
            print(f"Attaching {len(layers_to_hook)} extraction hooks:")
            for hook_name, layer_path in layers_to_hook.items():
                try:
                    transformer_module.add_extraction_hook(
                        submodule_path=layer_path,
                        hook_name=hook_name
                    )
                    self._hook_names.append(hook_name)
                    print(f"  ✓ {hook_name:30s} → {layer_path}")
                except Exception as e:
                    print(f"  ✗ {hook_name:30s} → {layer_path} (ERROR: {e})")
        
        print(f"{'='*70}\n")
        self._hooks_attached = True
    
    def manage_per_step_hooks(self, transformer_module: nn.Module, t: float, step_index: int):
        """Remove previous step's patch hooks and apply new ones for current step."""
        # Remove patch hooks from previous step
        for hook_name in self._active_patch_hooks:
            if hook_name in transformer_module.hook_handles:
                transformer_module.hook_handles[hook_name].remove()
                del transformer_module.hook_handles[hook_name]
                if hook_name in transformer_module.hook_objects:
                    del transformer_module.hook_objects[hook_name]
        self._active_patch_hooks.clear()

        # Add new patch hooks for this step
        patches_for_this_step = self.config.patches.get(step_index, [])
        
        if not patches_for_this_step:
            return

        for i, patch_config in enumerate(patches_for_this_step):
            layer_path = patch_config.get('path')
            patch_kwargs = patch_config.get('kwargs', {})
            
            if not layer_path:
                continue
                
            hook_name = f"__patch_step{step_index}_idx{i}"
            
            try:
                transformer_module.add_patch_hook(
                    submodule_path=layer_path,
                    hook_name=hook_name,
                    **patch_kwargs
                )
                self._active_patch_hooks.append(hook_name)
            except Exception as e:
                print(f"  [HookManager] ERROR adding patch '{hook_name}': {e}")

    def capture_and_save_timestep(self, transformer_module: nn.Module, t: float) -> bool:
        """
        Capture activations and optionally save to disk.
        
        CRITICAL: Only clear activations if configured to do so.
        For extraction experiments, activations must persist until after inference.
        """
        if not self._hooks_attached and not self.config.patches:
            return False
        
        step_index = self._step_counter
        self._step_counter += 1
        
        should_capture = self.config.should_capture_time(t, step_index)
        
        # Save to disk if requested
        if should_capture and self.config.save_per_timestep:
            activations = {}
            for hook_name in self._hook_names:
                if hook_name in transformer_module.extracted_activations:
                    captured_data = transformer_module.extracted_activations[hook_name]
                    activations[hook_name] = captured_data['tensor']
            
            if activations:
                self.config.save_timestep(t, step_index, activations)
        
        # Only clear if explicitly requested (not for extraction experiments)
        if self.config.clear_activations_per_step:
            transformer_module.clear_activations()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        return should_capture
    
    def cleanup(self, transformer_module: nn.Module):
        """
        Remove all hooks and save metadata.
        Does NOT clear activations - caller must access them first.
        """
        if self._hooks_attached or self._active_patch_hooks:
            transformer_module.remove_hooks()
            self._active_patch_hooks.clear()

            if self.config.save_per_timestep:
                self.config.save_metadata()
            
            print(f"\n{'='*70}")
            print(f"Capture complete!")
            print(f"  Total steps: {self._step_counter}")
            if self.config.save_per_timestep:
                print(f"  Files saved: {len(self.config.get_saved_files())}")
                print(f"  Directory: {self.config.save_dir}")
            else:
                num_activations = len(transformer_module.extracted_activations) if hasattr(transformer_module, 'extracted_activations') else 0
                print(f"  Activations in memory: {num_activations}")
            print(f"{'='*70}\n")
            
            self._hooks_attached = False

class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = None,
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = None,
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str, int] | None = None,
    ):
        super().__init__()
        
        odeint_kwargs = odeint_kwargs or dict(method="euler")
        mel_spec_kwargs = mel_spec_kwargs or dict()
        
        self.frac_lengths_mask = frac_lengths_mask
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.transformer = transformer
        self.dim = transformer.dim
        self.sigma = sigma
        self.odeint_kwargs = odeint_kwargs
        self.vocab_char_map = vocab_char_map
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    @torch.no_grad()
    def sample(
        self,
        cond,
        text,
        duration,
        *,
        lens=None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed=None,
        max_duration=4096,
        vocoder=None,
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
        
        # ODE function
        def fn(t, x):
            t_float = t.item() if t.ndim == 0 else t[0].item()
            
            if hook_manager:
                current_step_index = hook_manager._step_counter
                # Apply timestep-specific patches
                hook_manager.manage_per_step_hooks(self.transformer, t_float, current_step_index)
            
            # Predict flow
            if cfg_strength < 1e-5:
                pred = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=mask,
                    drop_audio_cond=False, drop_text=False, cache=True
                )
            else:
                pred_cfg = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=mask,
                    cfg_infer=True, cache=True
                )
                pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
                pred = pred + (pred - null_pred) * cfg_strength
            
            # Capture activations for this timestep
            if hook_manager:
                hook_manager.capture_and_save_timestep(self.transformer, t_float)
            
            return pred
        
        # Initial noise
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)
        
        # Time schedule
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
        
        # Run ODE
        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()
        
        # Cleanup hooks
        if hook_manager:
            hook_manager.cleanup(self.transformer)
        
        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)
        
        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)
        
        return out, trajectory
    
    def forward(self, inp, text, *, lens=None, noise_scheduler=None):
        """Training forward pass."""
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels
        
        batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device
        
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
            x=φ, cond=cond, text=text, time=time,
            drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask
        )
        
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]
        
        return loss.mean(), cond, pred