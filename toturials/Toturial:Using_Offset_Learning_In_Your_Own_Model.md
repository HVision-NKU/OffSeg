## How to Integrate Offset Learning into Your Own Model

This section provides a step-by-step guide on how to integrate Offset Learning into your own segmentation model. We use SegFormer and SegNeXt as examples to demonstrate the integration process.

### 🔧 Integration Overview

Offset Learning can be easily integrated into existing segmentation models by modifying the decode head to use the `Offset_Learning` module. The core idea is to replace the final classification layer with the offset learning mechanism for better spatial and class feature alignment.

### 📝 Step-by-Step Guide

#### Step 1: Understanding the Core Module

The `Offset_Learning` module is the core component that implements:
- **Dual offset learning**: Both class and feature offset learning
- **Coupled attention mechanism**: Computing attention between image features and class representations
- **Feature alignment**: Aligning spatial and class features for better segmentation

```python
from mmseg.models.decode_heads import Offset_Learning

# Core usage
offset_learning = Offset_Learning(
    num_classes=num_classes,    # Number of segmentation classes
    embed_dims=channels,        # Feature embedding dimensions
    init_std=0.02,             # Initialization standard deviation
    norm_cfg=dict(type='LN')   # Normalization configuration
)
```

#### Step 2: Create a Custom Decode Head

Here's how to create a custom decode head with Offset Learning for any backbone:

```python
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.models.decode_heads import Offset_Learning

@MODELS.register_module()
class YourModelHeadOffsetLearning(BaseDecodeHead):
    """Your custom model head with Offset Learning."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Your custom feature processing layers
        self.feature_proj = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        
        # Replace traditional conv_seg with Offset Learning
        # Remove the default classification head
        # delattr(self, 'conv_seg')
        
        # Add Offset Learning module
        self.offset_learning = Offset_Learning(
            num_classes=self.num_classes,
            embed_dims=self.channels
        )
    
    def forward(self, inputs):
        """Forward function."""
        # Transform inputs according to your model's needs
        x = self._transform_inputs(inputs)
        
        # Your custom feature processing
        x = self.feature_proj(x)
        
        # Apply Offset Learning instead of traditional classification
        output = self.offset_learning(x)
        
        return output
```

#### Step 3: Register Your Custom Module

Make sure to register your custom decode head in the MMSegmentation registry:

```python
# In  mmseg/models/decode_heads/__init__.py
from .yourmodelheadoffsetlearning import YourModelHeadOffsetLearning

__all__ = ['...', 'YourModelHeadOffsetLearning']
```

#### Step 4: Create Model Configuration

Create a configuration file for your model with Offset Learning:

```python
# your_model_offset_learning.py
_base_ = [
    '../_base_/datasets/ade20k.py',           # Dataset configuration
    '../_base_/default_runtime.py',           # Runtime configuration  
    '../_base_/schedules/schedule_160k.py'    # Training schedule
]

# Data preprocessing
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

# Model configuration
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='YourBackbone',
        # Your backbone configuration
    ),
    decode_head=dict(
        type='YourModelHeadOffsetLearning',    # Your custom head
        in_channels=[64, 128, 320, 512],       # Input channels from backbone
        in_index=[0, 1, 2, 3],                 # Feature indices to use
        channels=256,                          # Hidden channels
        num_classes=150,                       # Number of classes (ADE20K)
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0
        )
    ),
    train_cfg=dict(),
    test_cfg = dict(mode='slide', crop_size=(512, 512), stride=(480, 480))
    # or test_cfg=dict(mode='whole')
)

# Training configuration
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00006, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }
    )
)
```

### 🚀 Quick Start Examples

#### Example 1: SegFormer with Offset Learning

```bash
# Train SegFormer with Offset Learning
bash tools/dist_train.sh local_configs/segformer_offset_learning/B0/segformer_mit-b0_offset_learning_8xb2-160k_ade20k-512x512.py 8

# Evaluate
python tools/test.py local_configs/segformer_offset_learning/B0/segformer_mit-b0_offset_learning_8xb2-160k_ade20k-512x512.py /path/to/checkpoint.pth
```

#### Example 2: SegNeXt with Offset Learning  

```bash
# Train SegNeXt with Offset Learning
bash tools/dist_train.sh local_configs/segnext_offset_learning/Tiny/segnext_mscan-t_offset_learning_160k_ade20k-512x512.py

# Evaluate
python tools/test.py local_configs/segnext_offset_learning/Tiny/segnext_mscan-t_offset_learning_160k_ade20k-512x512.py /path/to/checkpoint.pth
```

### 🔍 Key Implementation Details

1. **Feature Requirements**: The input features should have shape `(B, C, H, W)` where `C` matches `embed_dims`
2. **Multi-scale Features**: For models with multi-scale features, fuse them before applying Offset Learning

For more examples and detailed implementations, refer to:
- `local_configs/segformer_offset_learning/` - SegFormer integration examples
- `local_configs/segnext_offset_learning/` - SegNeXt integration examples
- `mmseg/models/decode_heads/offset_learning.py` - Core Offset Learning implementation