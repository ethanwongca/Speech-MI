"""
DiT Model with Hooks

ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
from __future__ import annotations

print("\n\n<<<<<<<<<< Loading DiT Model with Hooks >>>>>>>>>>\n\n")

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


class FeatureExtractionHook:
    """Captures module output with proper memory management."""
    
    def __init__(self, storage: dict, name: str, output_index: int = 0):
        self.storage = storage
        self.name = name
        self.output_index = output_index
        self.call_count = 0
    
    def __call__(self, module, input, output):
        """
        Forward hook entry.
        Invokes PyTorch's forward hook, and saves the tensor
        """
        self.call_count += 1
        is_tuple = isinstance(output, tuple)
        target_tensor = output[self.output_index] if is_tuple else output
        with torch.no_grad():
            safe_tensor = target_tensor.detach().clone().cpu().to(torch.bfloat16)
        
        self.storage[self.name] = {
            'tensor': safe_tensor,
            'device': str(target_tensor.device),
            'shape': tuple(target_tensor.shape),
            'call_count': self.call_count
        }

class FeaturePatchHook:
    """Modifies and/or extracts activations during the forward pass."""

    def __init__(
        self,
        batch_index: int | None = None,
        feature_index: int | None = None,
        output_index: int = 0,
        patch_vec: torch.Tensor | float = 0.0,
        storage: dict | None = None,
        extract_name_before: str | None = None,
        extract_name_after: str | None = None,
    ):
        self.batch_index = batch_index
        self.feature_index = feature_index
        self.output_index = output_index
        self.patch_vec = patch_vec
        self.call_count = 0

        # Extraction storage
        self.storage = storage
        self.extract_name_before = extract_name_before
        self.extract_name_after = extract_name_after

    #  Helpers
    def _snapshot(self, tensor):
        """Return a detached, cloned, CPU-safe snapshot of an activation."""
        with torch.no_grad():
            # Fast GPU → CPU copy (async if pinned memory)
            return tensor.detach().clone().to("cpu", non_blocking=True).to(torch.bfloat16)

    def _replace_output(self, output, new_value):
        """Replace output_index inside tuple outputs."""
        if isinstance(output, tuple):
            return tuple(
                new_value if i == self.output_index else o
                for i, o in enumerate(output)
            )
        return new_value


    # Forward hook
    def __call__(self, module, inputs, outputs):
        """Forward hook entry point."""
        self.call_count += 1

        # Select the correct activation slice
        is_tuple = isinstance(outputs, tuple)
        target = outputs[self.output_index] if is_tuple else outputs

        # Extract BEFORE patching
        if self.storage is not None and self.extract_name_before:
            self.storage[self.extract_name_before] = {
                'tensor': self._snapshot(target),
                'shape': tuple(target.shape),
                'device': str(target.device),
                'call_count': self.call_count,
            }

        # Build detached clone for patching
        patched = target.detach().clone()

        # Move patch vector to correct device
        if isinstance(self.patch_vec, torch.Tensor):
            patch_vec = self.patch_vec.to(patched.device, non_blocking=True)
        else:
            patch_vec = self.patch_vec

        # ------------------------
        # Apply patch by dimension
        # ------------------------
        if patched.ndim == 3:
            self._patch_3d(patched, patch_vec)
        elif patched.ndim == 2:
            self._patch_2d(patched, patch_vec)
        else:
            # Extract after patching if requested (prevents error)
            if self.storage is not None and self.extract_name_after:
                self.storage[self.extract_name_after] = {
                    'tensor': self._snapshot(patched),
                    'shape': tuple(patched.shape),
                    'device': str(patched.device),
                    'call_count': self.call_count,
                }
            return outputs  
        # Extract after patching
        if self.storage is not None and self.extract_name_after:
            self.storage[self.extract_name_after] = {
                'tensor': self._snapshot(patched),
                'shape': tuple(patched.shape),
                'device': str(patched.device),
                'call_count': self.call_count,
            }

        return self._replace_output(outputs, patched)

    # Patching Operation Functions 

    def _patch_3d(self, tensor: torch.Tensor, vec):
    """Patch 3D tensors of shape (batch, seq, feature)."""
    
    _, S, _ = tensor.shape

    with torch.no_grad():
        if isinstance(vec, torch.Tensor):
            vec = vec.to(tensor.device, non_blocking=True)
            if vec.numel() == 1:
                vec = vec.item()
            else:
                vec = vec.reshape(-1)
                if vec.numel() != S:
                    raise ValueError(
                        f"Patch vector length {vec.numel()} must equal seq_len {S} or be scalar."
                    )

        if self.batch_index is not None and self.feature_index is not None:
            tensor[self.batch_index, :, self.feature_index] = vec
        elif self.batch_index is not None:
            tensor[self.batch_index, :, :] = vec
        elif self.feature_index is not None:
            tensor[:, :, self.feature_index] = vec
        else:
            tensor[:, :, :] = vec

    def _patch_2d(self, tensor: torch.Tensor, vec):
        """Patch 2D tensors of shape (batch, feature)."""

        with torch.no_grad():
            if self.batch_index is not None and self.feature_index is not None:
                tensor[self.batch_index, self.feature_index] = (
                    vec.item() if isinstance(vec, torch.Tensor) and vec.numel() == 1 else vec
                )
            elif self.batch_index is not None:
                tensor[self.batch_index, :] = vec
            elif self.feature_index is not None:
                tensor[:, self.feature_index] = vec
            else:
                tensor[:, :] = vec

# =======================================
# ORIGINAL DiT MODULES
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
    
    def forward(self, text: int, seq_len, drop_text=False, audio_mask=None):
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
    
    def forward(self, x, cond, text_embed, drop_audio_cond=False):
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# =======================================
# DiT BACKBONE
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
        self.text_embed = TextEmbedding(
            text_num_embeds, text_dim,
            mask_padding=text_mask_padding,
            average_upsampling=text_embedding_average_upsampling,
            conv_layers=conv_layers
        )
        self.text_cond, self.text_uncond = None, None
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.dim = dim
        self.depth = depth
        self.transformer_blocks = nn.ModuleList([
            DiTBlock(
                dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult,
                dropout=dropout, qk_norm=qk_norm, pe_attn_head=pe_attn_head,
                attn_backend=attn_backend, attn_mask_enabled=attn_mask_enabled
            )
            for _ in range(depth)
        ])
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)
        self.checkpoint_activations = checkpoint_activations
        self.initialize_weights()
        
        # Hook management
        self.hook_handles = {}
        self.hook_objects = {}
        self.extracted_activations = {}
    
    def initialize_weights(self):
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)
    
    def _get_submodule(self, path: str) -> nn.Module:
        """Navigate to submodule by path."""
        try:
            return self.get_submodule(path)
        except Exception as e:
            raise ValueError(f"Invalid submodule path '{path}': {e}")

    
    def add_extraction_hook(self, submodule_path: str, hook_name: str, output_index: int = 0):
    """Add hook to extract activations."""
    # Remove existing hook if present
    if hook_name in self.hook_handles:
        self.hook_handles[hook_name].remove()
        del self.hook_handles[hook_name]
        if hook_name in self.hook_objects:
            del self.hook_objects[hook_name]

    try:
        submodule = self._get_submodule(submodule_path)
        hook_obj = FeatureExtractionHook(self.extracted_activations, hook_name, output_index)

        handle = submodule.register_forward_hook(hook_obj, with_kwargs=True)
        self.hook_handles[hook_name] = handle
        self.hook_objects[hook_name] = hook_obj
    except Exception as e:
        print(f"ERROR adding extraction hook '{hook_name}' to '{submodule_path}': {e}")


    def add_patch_hook(self, submodule_path: str, hook_name: str, **patch_kwargs):
        """Add hook to patch activations."""
        # Remove existing hook if present
        if hook_name in self.hook_handles:
            self.hook_handles[hook_name].remove()
            del self.hook_handles[hook_name]
            if hook_name in self.hook_objects:
                del self.hook_objects[hook_name]

        try:
            submodule = self._get_submodule(submodule_path)

            # Inject shared storage dictionary
            patch_kwargs['storage'] = self.extracted_activations
            hook_obj = FeaturePatchHook(**patch_kwargs)
            handle = submodule.register_forward_hook(hook_obj, with_kwargs=True)

            self.hook_handles[hook_name] = handle
            self.hook_objects[hook_name] = hook_obj
        except Exception as e:
            print(f"ERROR adding patch hook '{hook_name}' to '{submodule_path}': {e}")


    def remove_hooks(self):
        """Remove all hooks and cleanup."""
        for handle in list(self.hook_handles.values()):
            handle.remove()
        self.hook_handles.clear()
        self.hook_objects.clear()


    def clear_activations(self):
        """Clear captured activations to free memory."""
        for key, entry in list(self.extracted_activations.items()):
            if isinstance(entry, dict) and 'tensor' in entry:
                del entry['tensor']
            del self.extracted_activations[key]
        self.extracted_activations.clear()


    def print_available_submodules(self, max_depth=3):
        """Print all hookable submodules."""
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
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            return module(*inputs)
        return ckpt_forward
    
    def get_input_embed(self, x, cond, text, drop_audio_cond=False, drop_text=False, cache=True, audio_mask=None):
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
    
    def forward(self, x, cond, text, time, mask=None, drop_audio_cond=False, drop_text=False, cfg_infer=False, cache=False):
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