# ==============================================================================
# f5_tts/model/backbones/dit.py (MODIFIED VERSION WITH FIXED HOOKS)
# ==============================================================================
from __future__ import annotations

# Diagnostic print to confirm this modified file is being loaded by Python
print("\n\n<<<<<<<<<< LOADING MODIFIED DiT BACKBONE WITH HOOKS >>>>>>>>>>\n\n")

import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    AdaLayerNorm_Final,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepEmbedding,
    get_pos_embed_indices,
    precompute_freqs_cis,
)

# =======================================
# HOOK CLASSES
# =======================================
class FeatureExtractionHook:
    """Creates a PyTorch forward hook to capture a module's output tensor."""
    def __init__(self, storage: dict, name: str, output_index: int = 0):
        self.storage = storage
        self.name = name
        self.output_index = output_index
        self.call_count = 0

    def __call__(self, module, input, output):
        self.call_count += 1
        is_tuple = isinstance(output, tuple)
        target_tensor = output[self.output_index] if is_tuple else output
        self.storage[self.name] = {
            'tensor': target_tensor.clone().detach().cpu().bfloat16(),
            'device': str(target_tensor.device),
            'shape': tuple(target_tensor.shape),
            'call_count': self.call_count
        }

class FeaturePatchHook:
    """Creates a PyTorch forward hook to modify (patch) a module's output tensor."""
    def __init__(self, batch_index: int | None = None, feature_index: int | None = None, 
                 output_index: int = 0, patch_vec: torch.Tensor | float = 0.0):
        self.batch_index = batch_index
        self.feature_index = feature_index
        self.output_index = output_index
        self.patch_vec = patch_vec
        self.call_count = 0

    def __call__(self, module, input, output):
        self.call_count += 1
        is_tuple = isinstance(output, tuple)
        target_tensor = output[self.output_index] if is_tuple else output
        patched_tensor = target_tensor.clone()

        # Ensure patch_vec is on the correct device
        if isinstance(self.patch_vec, torch.Tensor):
            patch_vec = self.patch_vec.to(patched_tensor.device)
        else:
            patch_vec = self.patch_vec

        if patched_tensor.ndim == 3:  # (batch, seq, feature)
            self._patch_3d(patched_tensor, patch_vec)
        elif patched_tensor.ndim == 2:  # (batch, feature)
            self._patch_2d(patched_tensor, patch_vec)
        else:
            return output

        if is_tuple:
            output_list = list(output)
            output_list[self.output_index] = patched_tensor
            return tuple(output_list)
        else:
            return patched_tensor

    def _patch_3d(self, tensor: torch.Tensor, patch_vec):
        if self.batch_index is not None and self.feature_index is not None:
            # Single element or broadcast
            if isinstance(patch_vec, torch.Tensor) and patch_vec.numel() > 1:
                # Ensure shape matches
                tensor[self.batch_index, :, self.feature_index] = patch_vec.reshape(-1)
            else:
                tensor[self.batch_index, :, self.feature_index] = patch_vec
        elif self.batch_index is not None:
            tensor[self.batch_index, :, :] = patch_vec
        elif self.feature_index is not None:
            tensor[:, :, self.feature_index] = patch_vec
        else:
            tensor[:, :, :] = patch_vec

    def _patch_2d(self, tensor: torch.Tensor, patch_vec):
        if self.batch_index is not None and self.feature_index is not None:
            if isinstance(patch_vec, torch.Tensor):
                tensor[self.batch_index, self.feature_index] = patch_vec.item() if patch_vec.numel() == 1 else patch_vec
            else:
                tensor[self.batch_index, self.feature_index] = patch_vec
        elif self.batch_index is not None:
            tensor[self.batch_index, :] = patch_vec
        elif self.feature_index is not None:
            tensor[:, self.feature_index] = patch_vec
        else:
            tensor[:, :] = patch_vec

# =======================================
# ORIGINAL DiT MODULES (Unchanged)
# =======================================
class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, mask_padding=True, average_upsampling=False, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)
        self.mask_padding = mask_padding
        self.average_upsampling = average_upsampling
        if average_upsampling:
            assert mask_padding, "text_embedding_average_upsampling requires text_mask_padding to be True"
        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(*[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)])
        else:
            self.extra_modeling = False

    def average_upsample_text_by_mask(self, text, text_mask, audio_mask):
        batch, text_len, text_dim = text.shape
        if audio_mask is None:
            audio_mask = torch.ones_like(text_mask, dtype=torch.bool)
        valid_mask = audio_mask & text_mask
        audio_lens = audio_mask.sum(dim=1)
        valid_lens = valid_mask.sum(dim=1)
        upsampled_text = torch.zeros_like(text)
        for i in range(batch):
            audio_len, valid_len = audio_lens[i].item(), valid_lens[i].item()
            if valid_len == 0: continue
            valid_ind = torch.where(valid_mask[i])[0]
            valid_data = text[i, valid_ind, :]
            base_repeat, remainder = audio_len // valid_len, audio_len % valid_len
            indices = []
            for j in range(valid_len):
                repeat_count = base_repeat + (1 if j >= valid_len - remainder else 0)
                indices.extend([j] * repeat_count)
            indices = torch.tensor(indices[:audio_len], device=text.device, dtype=torch.long)
            upsampled = valid_data[indices]
            upsampled_text[i, :audio_len, :] = upsampled
        return upsampled_text

    def forward(self, text: int, seq_len, drop_text=False, audio_mask: bool | None = None):
        text = text + 1
        text = text[:, :seq_len]
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)
        if self.mask_padding:
            text_mask = text == 0
        if drop_text:
            text = torch.zeros_like(text)
        text = self.text_embed(text)
        if self.extra_modeling:
            batch_start = torch.zeros((batch,), device=text.device, dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1), 0.0)
            else:
                text = self.text_blocks(text)
        if self.average_upsampling:
            text = self.average_upsample_text_by_mask(text, ~text_mask, audio_mask)
        return text

class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float, cond: float, text_embed: float, drop_audio_cond=False):
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x

# =======================================
# DiT BACKBONE WITH HOOK FUNCTIONALITY
# =======================================
class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        text_embedding_average_upsampling=False,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        attn_backend="torch",
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
    ):
        super().__init__()
        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None: text_dim = mel_dim
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, mask_padding=text_mask_padding, average_upsampling=text_embedding_average_upsampling, conv_layers=conv_layers)
        self.text_cond, self.text_uncond = None, None
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.dim = dim
        self.depth = depth
        self.transformer_blocks = nn.ModuleList([
            DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout, qk_norm=qk_norm, pe_attn_head=pe_attn_head, attn_backend=attn_backend, attn_mask_enabled=attn_mask_enabled)
            for _ in range(depth)
        ])
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)
        self.checkpoint_activations = checkpoint_activations
        self.initialize_weights()

        # --- Integrated Hook Management ---
        self.hook_handles = {}
        self.hook_objects = {}  # Store hook objects to access call_count
        self.extracted_activations = {}

    def initialize_weights(self):
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    # --- Hook Management Methods ---
    def _get_submodule(self, path: str) -> nn.Module:
        module = self
        for part in path.split('.'):
            if part.isdigit(): 
                module = module[int(part)]
            else: 
                module = getattr(module, part)
        return module

    def add_extraction_hook(self, submodule_path: str, hook_name: str, output_index: int = 0):
        if hook_name in self.hook_handles: 
            self.hook_handles[hook_name].remove()
        try:
            submodule = self._get_submodule(submodule_path)
            hook_obj = FeatureExtractionHook(self.extracted_activations, hook_name, output_index)
            handle = submodule.register_forward_hook(hook_obj)
            self.hook_handles[hook_name] = handle
            self.hook_objects[hook_name] = hook_obj
            print(f"Added extraction hook '{hook_name}' to '{submodule_path}'.")
        except Exception as e: 
            print(f"ERROR adding extraction hook '{hook_name}' to '{submodule_path}': {e}")

    def add_patch_hook(self, submodule_path: str, hook_name: str, **patch_kwargs):
        if hook_name in self.hook_handles: 
            self.hook_handles[hook_name].remove()
        try:
            submodule = self._get_submodule(submodule_path)
            hook_obj = FeaturePatchHook(**patch_kwargs)
            handle = submodule.register_forward_hook(hook_obj)
            self.hook_handles[hook_name] = handle
            self.hook_objects[hook_name] = hook_obj
            print(f"Added patch hook '{hook_name}' to '{submodule_path}'.")
        except Exception as e: 
            print(f"ERROR adding patch hook '{hook_name}' to '{submodule_path}': {e}")

    def remove_hooks(self):
        for handle in self.hook_handles.values(): 
            handle.remove()
        self.hook_handles.clear()
        self.hook_objects.clear()
        print("All hooks removed.")

    def clear_activations(self):
        self.extracted_activations.clear()

    def get_hook_stats(self):
        """Return statistics about hook execution."""
        stats = {
            'active_hooks': len(self.hook_handles),
            'captured_activations': len(self.extracted_activations),
            'hook_details': {}
        }
        for name, hook_obj in self.hook_objects.items():
            if isinstance(hook_obj, FeaturePatchHook):
                stats['hook_details'][name] = {
                    'type': 'patch',
                    'call_count': hook_obj.call_count
                }
            elif isinstance(hook_obj, FeatureExtractionHook):
                stats['hook_details'][name] = {
                    'type': 'extraction',
                    'call_count': hook_obj.call_count,
                    'captured': name in self.extracted_activations
                }
        return stats

    def print_available_submodules(self, max_depth=3):
        """Print all available submodules for hooking."""
        print("\n" + "="*70)
        print("AVAILABLE SUBMODULES FOR HOOKING")
        print("="*70)
        
        def recurse_print(module, prefix="", depth=0):
            if depth > max_depth:
                return
            for name, submodule in module.named_children():
                full_path = f"{prefix}.{name}" if prefix else name
                print(f"{'  ' * depth}├─ {name:30s} ({submodule.__class__.__name__})")
                recurse_print(submodule, full_path, depth + 1)
        
        recurse_print(self)
        print("="*70 + "\n")

    def __str__(self):
        report = "\n" + "="*70 + "\n"
        report += "CAPTURED ACTIVATIONS SUMMARY\n"
        report += "="*70 + "\n"
        if not self.extracted_activations: 
            return report + "  (No activations captured)\n" + "="*70 + "\n"
        for name, data in self.extracted_activations.items():
            tensor = data['tensor']
            report += f"  [{name}]\n"
            report += f"    Shape:      {data['shape']}\n"
            report += f"    Device:     {data['device']}\n"
            report += f"    Mean:       {tensor.mean():.6f}\n"
            report += f"    Std:        {tensor.std():.6f}\n"
            report += f"    Min:        {tensor.min():.6f}\n"
            report += f"    Max:        {tensor.max():.6f}\n"
            report += f"    Call Count: {data['call_count']}\n"
            report += "\n"
        return report + "="*70 + "\n"

    # --- Original Methods (Unchanged) ---
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            return module(*inputs)
        return ckpt_forward

    def get_input_embed(self, x, cond, text, drop_audio_cond: bool = False, drop_text: bool = False, cache: bool = True, audio_mask: bool | None = None):
        seq_len = x.shape[1]
        if cache:
            if drop_text:
                if self.text_uncond is None: 
                    self.text_uncond = self.text_embed(text, seq_len, drop_text=True, audio_mask=audio_mask)
                text_embed = self.text_uncond
            else:
                if self.text_cond is None: 
                    self.text_cond = self.text_embed(text, seq_len, drop_text=False, audio_mask=audio_mask)
                text_embed = self.text_cond
        else:
            text_embed = self.text_embed(text, seq_len, drop_text=drop_text, audio_mask=audio_mask)
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)
        return x

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    # --- Main Forward Pass (with hook integration) ---
    def forward(self, x: float, cond: float, text: int, time: float, mask: bool | None = None, drop_audio_cond: bool = False, drop_text: bool = False, cfg_infer: bool = False, cache: bool = False):
        self.clear_activations()  # Clear previous results
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)
        t = self.time_embed(time)
        if cfg_infer:
            x_cond = self.get_input_embed(x, cond, text, drop_audio_cond=False, drop_text=False, cache=cache, audio_mask=mask)
            x_uncond = self.get_input_embed(x, cond, text, drop_audio_cond=True, drop_text=True, cache=cache, audio_mask=mask)
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            if mask is not None: 
                mask = torch.cat((mask, mask), dim=0)
        else:
            x = self.get_input_embed(x, cond, text, drop_audio_cond=drop_audio_cond, drop_text=drop_text, cache=cache, audio_mask=mask)
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        if self.long_skip_connection is not None:
            residual = x
        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)
        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))
        x = self.norm_out(x, t)
        output = self.proj_out(x)
        return output

