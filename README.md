# Speech-MI: F5-TTS Activation Analysis

A hook system for extracting and patching activations in F5-TTS diffusion models.

## Setup

1. Clone and install F5-TTS as an editable library:
```bash
git clone https://github.com/SWivid/F5-TTS.git
cd F5-TTS
pip install -e .
```

2. Copy the `.py` files from this repository into the F5-TTS codebase at the appropriate locations.

## Features

- **Extract activations** from any layer during inference
- **Patch/modify activations** in real-time for ablation studies
- **Filter captures** by timestep or time value
- **Save data** to disk for analysis

## Quick Start

### Extract Activations

```python
from f5_tts.model.cfm import HookConfig, HookManager

# Configure what to capture
hook_config = HookConfig(
    capture_layers={
        'block_0': 'transformer_blocks.0',
        'block_11': 'transformer_blocks.11',
    },
    save_per_timestep=False,
    clear_activations_per_step=False
)

# Create manager and run inference
hook_manager = HookManager(hook_config)
audio, trajectory = cfm_model.sample(
    cond=cond,
    text=text,
    duration=duration,
    hook_manager=hook_manager,
    steps=32
)

# Access captured data
dit_model = cfm_model.transformer
for name, data in dit_model.extracted_activations.items():
    tensor = data['tensor']
    print(f"{name}: shape={tensor.shape}, mean={tensor.mean():.4f}")
```

### Patch Activations

```python
# Zero out a specific layer
hook_config = HookConfig(
    patches={
        'zero_block_5': {
            'path': 'transformer_blocks.5',
            'kwargs': {'patch_vec': 0.0}
        }
    }
)

hook_manager = HookManager(hook_config)
audio, trajectory = cfm_model.sample(
    cond=cond,
    text=text,
    duration=duration,
    hook_manager=hook_manager,
    steps=32
)
```

## HookConfig Options

```python
HookConfig(
    capture_layers: dict[str, str] = {},      # {name: layer_path}
    capture_all_layers: bool = False,          # Capture all transformer blocks
    time_filter: Callable = None,              # Filter which timesteps to capture
    save_per_timestep: bool = True,            # Save each timestep to disk
    save_dir: str = "./hook_captures",         # Directory for saved captures
    patches: dict = {},                        # Patch configurations
    clear_activations_per_step: bool = True,   # Clear memory after each step
    precision: torch.dtype = torch.bfloat16,   # Dtype for saved tensors
    device: str = 'cpu'                        # Device for saved tensors
)
```

## Available Layer Paths

```
input_embed                  # Input embedding layer
text_embed                   # Text embedding layer
time_embed                   # Timestep embedding layer
transformer_blocks.0         # First transformer block
transformer_blocks.N         # Nth transformer block
norm_out                     # Final normalization
proj_out                     # Output projection
```

Find available paths with:
```python
dit_model.print_available_submodules(max_depth=3)
```

## Time Filtering

### Capture every Nth timestep
```python
def time_filter_every_nth(n: int):
    return lambda t, step: step % n == 0

hook_config = HookConfig(
    capture_layers={'block_0': 'transformer_blocks.0'},
    time_filter=time_filter_every_nth(4)
)
```

### Capture specific steps
```python
def time_filter_specific_steps(steps: list[int]):
    steps_set = set(steps)
    return lambda t, step: step in steps_set

hook_config = HookConfig(
    capture_layers={'block_0': 'transformer_blocks.0'},
    time_filter=time_filter_specific_steps([0, 15, 31])
)
```

## Saving and Loading

### Save to disk
```python
hook_config = HookConfig(
    capture_layers={'block_0': 'transformer_blocks.0'},
    save_per_timestep=True,
    save_dir="./my_captures"
)
```

### Load saved data
```python
timesteps = HookConfig.load_from_dir("./my_captures")
for ts_data in timesteps:
    t = ts_data['t']
    step = ts_data['step_index']
    activations = ts_data['activations']
```

## Use Cases

- Ablation studies
- Activation analysis
- Model interpretability
- Debugging diffusion dynamics
- Feature engineering experiments
- Feature engineering experiments

For more examples, see the ablation study script in the repository.

