# Speech MI F5-TTS

1. Create F5-TTS local editable library 
```
git clone https://github.com/SWivid/F5-TTS.git
cd F5-TTS
pip install -e .
```

2. Replace the DIT.py, CFM.py, and utils_infer.py with the code files within the repository. 

# Using the Code 

## Overview

The modified DiT (Diffusion Transformer) and CFM (Continuous Flow Matching) modules provide a powerful hook system for:
- **Extracting activations** from any layer during inference
- **Patching/modifying activations** in real-time (ablation studies)
- **Saving timestep data** to disk for analysis
- **Filtering captures** by timestep or time value

This system enables deep model introspection, ablation studies, and activation engineering without modifying the core model architecture.

---

## Architecture

### Components

1. **DiT (Diffusion Transformer)** - `f5_tts/model/backbones/dit.py`
   - Contains the transformer backbone
   - Provides hook attachment methods
   - Stores extracted activations

2. **CFM (Continuous Flow Matching)** - `f5_tts/model/cfm.py`
   - Orchestrates the ODE sampling process
   - Integrates `HookManager` into the sampling loop
   - Coordinates timestep captures

3. **HookConfig** - Configuration object
   - Defines what to capture/patch
   - Controls filtering and saving behavior

4. **HookManager** - Orchestration layer
   - Sets up hooks on DiT model
   - Manages capture timing
   - Handles cleanup

---

## Quick Start

### Basic Extraction Example

```python
from f5_tts.model.cfm import HookConfig, HookManager

# 1. Define what to capture
hook_config = HookConfig(
    capture_layers={
        'block_0': 'transformer_blocks.0',
        'block_11': 'transformer_blocks.11',
        'final_out': 'proj_out'
    },
    save_per_timestep=False,  # Keep in memory only
    clear_activations_per_step=False  # Preserve final values
)

# 2. Create manager
hook_manager = HookManager(hook_config)

# 3. Pass to inference
audio, trajectory = cfm_model.sample(
    cond=cond,
    text=text,
    duration=duration,
    hook_manager=hook_manager,  # <-- Add this
    steps=32
)

# 4. Access captured data
dit_model = cfm_model.transformer
for name, data in dit_model.extracted_activations.items():
    tensor = data['tensor']
    print(f"{name}: shape={tensor.shape}, mean={tensor.mean():.4f}")
```

### Basic Patching Example

```python
# Zero out a specific layer
hook_config = HookConfig(
    patches={
        'zero_block_5': {
            'path': 'transformer_blocks.5',
            'kwargs': {
                'patch_vec': 0.0,
                'batch_index': None,
                'feature_index': None
            }
        }
    }
)

hook_manager = HookManager(hook_config)

# Run inference with the patch applied
audio, trajectory = cfm_model.sample(
    cond=cond,
    text=text,
    duration=duration,
    hook_manager=hook_manager,
    steps=32
)
```

---

## HookConfig Reference

### Constructor Parameters

```python
HookConfig(
    capture_all_layers: bool = False,
    capture_layers: dict[str, str] = {},
    capture_final_output: bool = False,
    time_filter: Callable[[float, int], bool] | None = None,
    precision: torch.dtype = torch.bfloat16,
    device: str = 'cpu',
    save_per_timestep: bool = True,
    save_dir: str | Path = "./hook_captures",
    patches: dict | None = {},
    clear_activations_per_step: bool = True
)
```

#### Parameters Explained

- **`capture_all_layers`**: If `True`, automatically captures all transformer blocks
- **`capture_layers`**: Dict of `{hook_name: layer_path}` for specific layers
- **`capture_final_output`**: Captures the final projection output
- **`time_filter`**: Function `(t: float, step: int) -> bool` to filter timesteps
- **`precision`**: Dtype for saved tensors (default: `bfloat16`)
- **`device`**: Device for saved tensors (default: `'cpu'`)
- **`save_per_timestep`**: Save each timestep to disk
- **`save_dir`**: Directory for saved captures
- **`patches`**: Dict of patch configurations (see below)
- **`clear_activations_per_step`**: Clear memory after each step (set `False` for extraction)

---

## Layer Path Reference

### Available Paths in DiT

```
input_embed              # Input embedding layer
input_embed.proj         # Linear projection
input_embed.conv_pos_embed  # Convolutional position embedding

text_embed               # Text embedding layer
time_embed               # Timestep embedding layer

transformer_blocks.0     # First transformer block
transformer_blocks.1     # Second transformer block
...
transformer_blocks.23    # Last block (assuming 24 blocks)

norm_out                 # Final normalization
proj_out                 # Output projection
```

### Finding Available Paths

```python
# Print all available submodules
dit_model = cfm_model.transformer
dit_model.print_available_submodules(max_depth=3)
```

---

## Time Filtering

Control which timesteps to capture during the ODE solve.

### Filter All Timesteps

```python
def time_filter_all():
    return lambda t, step: True

hook_config = HookConfig(
    capture_layers={'block_0': 'transformer_blocks.0'},
    time_filter=time_filter_all()
)
```

### Filter Every Nth Timestep

```python
def time_filter_every_nth(n: int):
    return lambda t, step: step % n == 0

# Capture every 4th timestep
hook_config = HookConfig(
    capture_layers={'block_0': 'transformer_blocks.0'},
    time_filter=time_filter_every_nth(4)
)
```

### Filter Specific Steps

```python
def time_filter_specific_steps(steps: list[int]):
    steps_set = set(steps)
    return lambda t, step: step in steps_set

# Capture only steps 0, 15, and 31
hook_config = HookConfig(
    capture_layers={'block_0': 'transformer_blocks.0'},
    time_filter=time_filter_specific_steps([0, 15, 31])
)
```

### Filter by Time Range

```python
def time_filter_time_range(t_min: float, t_max: float):
    return lambda t, step: t_min <= t <= t_max

# Capture only middle portion of diffusion (t=0.3 to t=0.7)
hook_config = HookConfig(
    capture_layers={'block_0': 'transformer_blocks.0'},
    time_filter=time_filter_time_range(0.3, 0.7)
)
```

### Filter Endpoints

```python
def time_filter_endpoints(early: float = 0.2, late: float = 0.8):
    return lambda t, step: t < early or t > late

# Capture early (t<0.2) and late (t>0.8) timesteps
hook_config = HookConfig(
    capture_layers={'block_0': 'transformer_blocks.0'},
    time_filter=time_filter_endpoints(0.2, 0.8)
)
```

---

## Patching (Ablation)

### Patch Configuration Structure

```python
patches = {
    'patch_name': {
        'path': 'layer_path',
        'kwargs': {
            'batch_index': int | None,
            'feature_index': int | None,
            'output_index': int,
            'patch_vec': torch.Tensor | float
        }
    }
}
```

### Zero Out Entire Layer

```python
hook_config = HookConfig(
    patches={
        'zero_block_0': {
            'path': 'transformer_blocks.0',
            'kwargs': {'patch_vec': 0.0}
        }
    }
)
```

### Zero Out Specific Feature Dimension

```python
hook_config = HookConfig(
    patches={
        'zero_feature_10': {
            'path': 'transformer_blocks.0',
            'kwargs': {
                'feature_index': 10,  # Zero only dimension 10
                'patch_vec': 0.0
            }
        }
    }
)
```

### Constant Imputation

```python
hook_config = HookConfig(
    patches={
        'constant_block': {
            'path': 'transformer_blocks.5',
            'kwargs': {'patch_vec': 0.5}  # Set to constant 0.5
        }
    }
)
```

### Reinsert Captured Activation

```python
# Step 1: Capture activation from one run
capture_config = HookConfig(
    capture_layers={'block_0': 'transformer_blocks.0'},
    save_per_timestep=False,
    clear_activations_per_step=False
)
hook_manager = HookManager(capture_config)
audio, _ = cfm_model.sample(..., hook_manager=hook_manager)

# Step 2: Extract the tensor
captured = dit_model.extracted_activations['block_0']['tensor']

# Step 3: Reinsert into different layer
patch_config = HookConfig(
    patches={
        'reinsert_to_block_10': {
            'path': 'transformer_blocks.10',
            'kwargs': {'patch_vec': captured}
        }
    }
)
hook_manager = HookManager(patch_config)
audio, _ = cfm_model.sample(..., hook_manager=hook_manager)
```

### Multiple Patches

```python
hook_config = HookConfig(
    patches={
        'zero_block_0': {
            'path': 'transformer_blocks.0',
            'kwargs': {'patch_vec': 0.0}
        },
        'zero_block_1': {
            'path': 'transformer_blocks.1',
            'kwargs': {'patch_vec': 0.0}
        },
        'const_text': {
            'path': 'text_embed',
            'kwargs': {'patch_vec': 0.1}
        }
    }
)
```

---

## Saving to Disk

### Enable Per-Timestep Saving

```python
hook_config = HookConfig(
    capture_layers={'block_0': 'transformer_blocks.0'},
    save_per_timestep=True,
    save_dir="./my_captures",
    time_filter=time_filter_every_nth(4)  # Save every 4th step
)

hook_manager = HookManager(hook_config)
audio, _ = cfm_model.sample(..., hook_manager=hook_manager)

# Files saved to ./my_captures/timestep_0000_t_0.000000.pt, etc.
```

### Loading Saved Data

```python
# Load all timesteps from a directory
timesteps = HookConfig.load_from_dir("./my_captures")

for ts_data in timesteps:
    t = ts_data['t']
    step = ts_data['step_index']
    activations = ts_data['activations']
    
    for hook_name, data in activations.items():
        tensor = data['tensor']
        print(f"Step {step}, t={t:.3f}, {hook_name}: {tensor.shape}")
```

### Metadata File

When saving, a `capture_metadata.json` is created:

```json
{
  "precision": "torch.bfloat16",
  "device": "cpu",
  "capture_all_layers": false,
  "capture_layers": ["block_0", "block_11"],
  "timesteps": [
    {
      "step_index": 0,
      "t": 0.0,
      "file": "timestep_0000_t_0.000000.pt",
      "num_hooks": 2
    }
  ]
}
```

---

## Complete Examples

### Example 1: Extract Activations from Multiple Layers

```python
from f5_tts.model.cfm import HookConfig, HookManager

# Configure extraction
config = HookConfig(
    capture_layers={
        'input': 'input_embed',
        'block_0': 'transformer_blocks.0',
        'block_mid': 'transformer_blocks.11',
        'block_last': 'transformer_blocks.23',
        'output': 'proj_out'
    },
    save_per_timestep=False,
    clear_activations_per_step=False  # Keep final values
)

manager = HookManager(config)

# Run inference
audio, trajectory = cfm_model.sample(
    cond=mel_spec,
    text=text_tokens,
    duration=target_duration,
    hook_manager=manager,
    steps=32
)

# Analyze captures
dit = cfm_model.transformer
for name, data in dit.extracted_activations.items():
    tensor = data['tensor']
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Mean: {tensor.mean():.6f}")
    print(f"  Std: {tensor.std():.6f}")
    print(f"  Range: [{tensor.min():.6f}, {tensor.max():.6f}]")
```

### Example 2: Progressive Ablation Study

```python
# Test each transformer block individually
results = {}

for block_idx in range(24):  # Assuming 24 blocks
    config = HookConfig(
        patches={
            f'zero_block_{block_idx}': {
                'path': f'transformer_blocks.{block_idx}',
                'kwargs': {'patch_vec': 0.0}
            }
        }
    )
    
    manager = HookManager(config)
    
    audio, _ = cfm_model.sample(
        cond=mel_spec,
        text=text_tokens,
        duration=target_duration,
        hook_manager=manager,
        steps=32
    )
    
    results[f'block_{block_idx}_zeroed'] = audio

# Compare audio quality across ablations
```

### Example 3: Capture Diffusion Trajectory

```python
# Capture activations at key diffusion timesteps
config = HookConfig(
    capture_layers={'block_11': 'transformer_blocks.11'},
    time_filter=time_filter_specific_steps([0, 8, 16, 24, 31]),
    save_per_timestep=True,
    save_dir="./diffusion_trajectory"
)

manager = HookManager(config)
audio, _ = cfm_model.sample(
    cond=mel_spec,
    text=text_tokens,
    duration=target_duration,
    hook_manager=manager,
    steps=32
)

# Analyze how activations evolve
timesteps = HookConfig.load_from_dir("./diffusion_trajectory")
for ts in timesteps:
    tensor = ts['activations']['block_11']['tensor']
    print(f"t={ts['t']:.3f}: mean={tensor.mean():.4f}, std={tensor.std():.4f}")
```

### Example 4: Combined Extraction and Patching

```python
# Extract from one layer, zero out another
config = HookConfig(
    capture_layers={
        'monitor_block_0': 'transformer_blocks.0'
    },
    patches={
        'zero_block_5': {
            'path': 'transformer_blocks.5',
            'kwargs': {'patch_vec': 0.0}
        }
    },
    save_per_timestep=False,
    clear_activations_per_step=False
)

manager = HookManager(config)
audio, _ = cfm_model.sample(
    cond=mel_spec,
    text=text_tokens,
    duration=target_duration,
    hook_manager=manager,
    steps=32
)

# Check if block_0 compensates for zeroed block_5
block_0_tensor = dit_model.extracted_activations['monitor_block_0']['tensor']
print(f"Block 0 activation stats: mean={block_0_tensor.mean():.4f}")
```

---

## Tips and Best Practices

### Memory Management

1. **Set `clear_activations_per_step=True`** when saving to disk to avoid memory buildup
2. **Set `clear_activations_per_step=False`** when extracting final values only
3. Use `time_filter` to reduce the number of captures

### Debugging

```python
# Check hook installation
stats = dit_model.get_hook_stats()
print(f"Active hooks: {stats['active_hooks']}")
print(f"Captured activations: {stats['captured_activations']}")
for name, info in stats['hook_details'].items():
    print(f"  {name}: {info['type']}, calls={info['call_count']}")
```

### Finding Layer Paths

```python
# List all available submodules
dit_model.print_available_submodules(max_depth=3)

# Access a specific submodule to inspect
module = dit_model._get_submodule('transformer_blocks.0')
print(module)
```

### Performance Considerations

- **Extraction**: Minimal overhead (~1-5% slowdown)
- **Patching**: Negligible overhead
- **Disk saving**: Can add 10-30% overhead depending on capture frequency
- Use `time_filter_every_nth(n)` to reduce saves

---

## Troubleshooting

### "Hook not found" Error

```python
# Check available paths first
dit_model.print_available_submodules()

# Verify the path exists
try:
    module = dit_model._get_submodule('your_path_here')
    print("Path is valid!")
except Exception as e:
    print(f"Invalid path: {e}")
```

### No Activations Captured

```python
# Make sure clear_activations_per_step=False for extraction
config = HookConfig(
    capture_layers={'block_0': 'transformer_blocks.0'},
    clear_activations_per_step=False  # <-- Important!
)
```

### Shape Mismatch in Patching

```python
# First capture to see the shape
# Then ensure patch_vec matches
captured = dit_model.extracted_activations['block_0']['tensor']
print(f"Expected shape: {captured.shape}")

# Use the captured tensor as patch_vec
config = HookConfig(
    patches={
        'patch': {
            'path': 'transformer_blocks.0',
            'kwargs': {'patch_vec': captured}
        }
    }
)
```

---

## Advanced: Custom Hook Functions

While the system provides `FeatureExtractionHook` and `FeaturePatchHook`, you can create custom hooks:

```python
class CustomHook:
    def __init__(self, storage, name):
        self.storage = storage
        self.name = name
    
    def __call__(self, module, input, output):
        # Custom processing
        processed = output.mean(dim=1)  # Example: average over sequence
        self.storage[self.name] = processed.detach().cpu()

# Register manually
hook_obj = CustomHook(dit_model.extracted_activations, 'custom')
handle = dit_model.transformer_blocks[0].register_forward_hook(hook_obj)
```

---

## Summary

The DiT and CFM hook system provides:

- Non-invasive activation extraction
- Real-time activation patching
- Timestep-aware filtering
- Disk persistence for large studies
- Flexible configuration
- Zero changes to model architecture

Use this system for:
- Ablation studies
- Activation analysis
- Model interpretability
- Debugging diffusion dynamics
- Feature engineering experiments

For more examples, see the ablation study script in the repository.

