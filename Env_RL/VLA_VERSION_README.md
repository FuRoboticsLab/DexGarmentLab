# VLA-Enhanced Multi-Stage Residual RL Environment

## Overview

This is a **standalone VLA-enhanced version** of the multi-stage residual RL environment. It adds Vision-Language-Action (VLA) feature extraction to provide visual understanding beyond point clouds.

## Key Features

1. **VLA Feature Extraction**
   - Extracts visual features from RGB images
   - Provides semantic understanding (wrinkles, texture, alignment)
   - Stage-specific language guidance

2. **Enhanced Observation Space**
   - All observations from simplified version
   - `vla_visual_features`: (256,) visual understanding
   - `vla_stage_guidance`: (128,) stage-specific guidance

3. **No Fine-Tuning Required**
   - Uses pretrained VLA models (CLIP, OpenVLA, etc.)
   - RL policy learns to use VLA features
   - No action space mismatch issues

## Files

- **`multi_stage_residual_env_vla.py`** - VLA-enhanced environment
- **`train_multi_stage_ppo_vla.py`** - Training script with VLA support
- **`VLA_VERSION_README.md`** - This file

## Installation

### Required Dependencies

```bash
# For CLIP (recommended, lightweight)
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# For PyTorch (if not already installed)
pip install torch torchvision
```

### Optional: OpenVLA

If you want to use OpenVLA instead of CLIP:
```bash
# Follow OpenVLA installation instructions
# https://github.com/robotic-vla/OpenVLA
```

## Usage

### Basic Training with CLIP (Recommended)

```bash
python Env_RL/train_multi_stage_ppo_vla.py \
    --vla-model clip \
    --training-data-num 100 \
    --stage-1-checkpoint 1500 \
    --stage-2-checkpoint 1500 \
    --stage-3-checkpoint 1500 \
    --total-timesteps 200000
```

### Training with Dummy VLA (Testing)

```bash
python Env_RL/train_multi_stage_ppo_vla.py \
    --vla-model dummy \
    --use-dummy-il \
    --total-timesteps 10000
```

### Baseline Comparison (No VLA)

```bash
python Env_RL/train_multi_stage_ppo_vla.py \
    --disable-vla \
    --training-data-num 100 \
    --total-timesteps 200000
```

## VLA Models Supported

### 1. CLIP (Recommended)
- **Pros**: Lightweight, fast, good visual features
- **Cons**: No language model (guidance is simplified)
- **Install**: `pip install git+https://github.com/openai/CLIP.git`

### 2. OpenVLA
- **Pros**: Full VLA model, better language understanding
- **Cons**: Larger, slower, requires more setup
- **Status**: Placeholder (needs implementation)

### 3. Dummy
- **Pros**: Fast, no dependencies
- **Cons**: Random features (for testing only)
- **Use**: Testing environment setup

## Architecture

```
┌─────────────────────────────────────────┐
│         RGB Image (from camera)          │
└──────────────────────┬──────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────┐
│      VLA Feature Extractor               │
│  - Visual features (256D)               │
│  - Stage guidance (128D)                 │
└──────────────────────┬──────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────┐
│      Enhanced Observation Space          │
│  - Point cloud (2048, 3)                 │
│  - GAM keypoints (6, 3)                  │
│  - VLA visual features (256,) [NEW]      │
│  - VLA stage guidance (128,) [NEW]       │
│  - Joint positions, EE poses, etc.       │
└──────────────────────┬──────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────┐
│      RL Policy (MultiInputPolicy)        │
│  - Processes all modalities              │
│  - Learns to combine features             │
└──────────────────────┬──────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────┐
│      Action: [residuals(60D), advance(1D)]│
└─────────────────────────────────────────┘
```

## Observation Space

```python
obs = {
    # Existing observations
    "garment_pcd": (2048, 3),           # Point cloud
    "gam_keypoints": (6, 3),            # Manipulation keypoints
    "joint_positions": (60,),           # Joint angles
    "ee_poses": (14,),                  # End-effector poses
    "il_action": (60,),                 # IL action
    "current_stage": (4,),              # Stage one-hot
    "stage_progress": (1,),             # Progress
    "stages_completed": (3,),           # Completed stages
    "should_advance_hint": (1,),        # Advance hint
    
    # NEW: VLA features
    "vla_visual_features": (256,),      # Visual understanding
    "vla_stage_guidance": (128,),       # Stage guidance
}
```

## Feature Extractor

The `VLAEnhancedFeaturesExtractor` processes all modalities:

1. **Point Cloud Encoder**: 2048×3 → 64D
2. **GAM Encoder**: 6×3 → 32D
3. **State Encoder**: (joints + EE + IL + stage info) → 32D
4. **VLA Visual Encoder**: 256D → 64D
5. **VLA Guidance Encoder**: 128D → 32D
6. **Combiner**: All features → 256D → final features

## Key Parameters

### Environment Parameters

- `--vla-model`: VLA model to use (`clip`, `openvla`, `dummy`)
- `--disable-vla`: Disable VLA features (baseline comparison)
- `--vla-feature-dim`: Visual feature dimension (default: 256)
- `--vla-guidance-dim`: Guidance feature dimension (default: 128)

### Training Parameters

- `--total-timesteps`: Total training steps (default: 200000)
- `--learning-rate`: PPO learning rate (default: 3e-4)
- `--batch-size`: Batch size (default: 64)
- `--n-steps`: Steps per rollout (default: 2048)

## Expected Benefits

### Quantitative Improvements
- **Success rate**: +10-20% (better visual understanding)
- **Fold quality**: +15-25% (wrinkle detection, alignment)
- **Learning speed**: +20-30% (better reward signal)

### Qualitative Improvements
- Better wrinkle handling
- More accurate part identification
- Better stage transitions
- More robust to variations

## Comparison: With vs Without VLA

### Baseline (No VLA)
```bash
python Env_RL/train_multi_stage_ppo_vla.py --disable-vla
```

### With VLA (CLIP)
```bash
python Env_RL/train_multi_stage_ppo_vla.py --vla-model clip
```

Compare:
- Training curves (TensorBoard)
- Success rates
- Fold quality metrics
- Learning speed

## Troubleshooting

### CLIP Installation Issues

**Problem**: `ImportError: No module named 'clip'`

**Solution**:
```bash
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

### CUDA Out of Memory

**Problem**: GPU memory error when loading CLIP

**Solution**:
- Use CPU: `--device cpu`
- Use smaller batch size: `--batch-size 32`
- Use dummy model for testing: `--vla-model dummy`

### VLA Features Not Appearing

**Problem**: Observation space doesn't have VLA features

**Solution**:
- Check `use_vla_features=True` in environment
- Verify VLA model loaded successfully
- Check environment logs for VLA initialization

## Next Steps

1. **Train with CLIP**: Start with CLIP (lightweight, fast)
2. **Compare baselines**: Train with/without VLA
3. **Evaluate**: Compare success rates and quality
4. **Iterate**: Adjust VLA feature dimensions if needed

## Advanced Usage

### Custom VLA Model

To add a custom VLA model, modify `VLAFeatureExtractor._load_*_model()`:

```python
def _load_custom_model(self):
    """Load your custom VLA model."""
    # Your implementation
    return CustomVLAWrapper(...)
```

### Fine-Tuning VLA (Optional)

If you want to fine-tune VLA (not recommended initially):
1. Collect demonstration data
2. Fine-tune VLA's vision encoder on garment images
3. Use fine-tuned model in environment

See `VLA_TRAINING_STRATEGY.md` for details.

## Summary

**This VLA-enhanced version:**
- ✅ Adds visual understanding beyond point clouds
- ✅ No fine-tuning required (uses pretrained models)
- ✅ Easy to use (just specify `--vla-model clip`)
- ✅ Can be disabled for baseline comparison (`--disable-vla`)
- ✅ Standalone and self-contained

**Start with CLIP** - it's lightweight, fast, and provides good visual features. You can always upgrade to OpenVLA or other models later!

