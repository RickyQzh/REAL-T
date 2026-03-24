# Local Model Adaptation Guide

This document explains how to adapt any TSE (Target Speaker Extraction) model to Wesep's `load_model_local` function.

## Framework Note

The inference and evaluation are **separated** in this framework:

- If you only want to use your own model for inference, you can organize the inference results in the format required by evaluation
- If you want to adapt your model to the current framework (wesep) inference format, please refer to the guide and examples below

## Directory Structure Requirements

```
local_model/
├── config.yaml          # Model configuration file
├── avg_model.pt         # Model weights file
└── [other model files]  # e.g., speaker model, pre-trained models, etc.
```

## config.yaml Format

The configuration file must contain the following fields:

```yaml
model:
  tse_model: ModelClassName  # e.g., ConvTasNet, BSRNN, etc.

model_args:
  tse_model:
    # Model initialization parameters, depends on specific model
    sr: 16000              # Sample rate
    win: 512               # Window size
    stride: 128            # Stride
    feature_dim: 128       # Feature dimension
    joint_training: true   # Must be true
    # Other model-specific parameters...

dataset_args:
  resample_rate: 16000    # Output sample rate
```

**Note:** You must set `model_args.tse_model.joint_training: true`, otherwise inference will fail.

## Adaptation Steps

### 1. Register Your Model

If using a custom model, register it in `wesep/models/__init__.py`:

```python
# Add import
import your_model_path as your_module_name

# In get_model function, add:
elif model_name.startswith("Your_Model_Prefix"):
    return getattr(your_module_name, model_name)
```

### 2. Prepare Model Weights

The model checkpoint needs to be converted to the required format:

1. Original training checkpoint format:
   ```python
   {
       'epoch': ...,
       'model_state_dict': model_weights,
       'optim_state_dict': ...,
       ...
   }
   ```

2. Convert to inference format:
   ```python
   {'models': [model_weights]}
   ```

Conversion script example:
```python
import torch

# Load original checkpoint
ckpt = torch.load('your_checkpoint.pt', map_location='cpu')

# Extract model weights (adjust key based on original format)
model_state = ckpt['model_state_dict']

# Convert to inference format
new_ckpt = {'models': [model_state]}

# Save as avg_model.pt
torch.save(new_ckpt, 'avg_model.pt')
```

### 3. Configure Parameters

Add the corresponding configuration items in `config.yaml` based on your model's `__init__` parameters.

**Note:** You must set `model_args.tse_model.joint_training: true`, otherwise inference will fail.

### 4. Other Dependencies

If your model uses additional pre-trained models (e.g., speaker model), ensure these files are also in the `local_model` directory and correctly referenced in the configuration.

## Usage

Modify the model loading code in `tse_baseline`, change from:

```python
model = wesep.load_model("english")
```

to:

```python
model = wesep.load_model_local("your/local_model")
```

Then run the `run_tse.sh` script:

```bash
bash run_tse.sh
```

## Notes

- Some parameters may be automatically overridden during inference (e.g., speaker model initialization), refer to the specific model code
- Ensure the model weights file format is compatible with the loading code
- The model class name must start with a prefix registered in `get_model`
