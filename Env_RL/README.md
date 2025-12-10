# DexGarmentLab RL Module

This module provides reinforcement learning capabilities for garment manipulation tasks, building on top of the DexGarmentLab simulation environment and HALO imitation learning policies.

## Overview

The RL module offers **four distinct approaches** for training garment manipulation policies, ranging from basic PPO to sophisticated IL+RL hybrid systems:

| Approach | Description | Action Space | Best For |
|----------|-------------|--------------|----------|
| **Basic PPO** | Standard RL from scratch | 8D (EE deltas + grippers) | Baselines, simple tasks |
| **Hierarchical PPO** | RL selects IK primitives | 6D discrete | Sequence learning |
| **Hierarchical IL PPO** | RL selects SADP_G stages | 6D discrete | Leveraging trained IL |
| **Multi-Stage Residual** ⭐ | IL + RL corrections | 61D (joint residuals) | **Best performance** |

## Installation

Install the additional RL dependencies:

```bash
pip install gymnasium stable-baselines3[extra] tensorboard opencv-python
```

## Quick Start

### Basic PPO Training

Train a PPO agent on the Fold Tops task from scratch:

```bash
# With GUI (for debugging)
python Env_RL/train_ppo.py --total-timesteps 500000

# Headless mode (faster training)
python Env_RL/train_ppo.py --headless --total-timesteps 1000000

# With custom garment
python Env_RL/train_ppo.py --garment-usd "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Jacket032/TCLC_Jacket032_obj.usd"
```

### Multi-Stage Residual RL (Recommended)

The best-performing approach that combines SADP_G with learned corrections:

```bash
# Test with dummy IL policy (no checkpoints needed)
python Env_RL/train_multi_stage_ppo.py --use-dummy-il --headless

# With actual SADP_G checkpoints (recommended)
python Env_RL/train_multi_stage_ppo.py \
    --training-data-num 100 \
    --stage-1-checkpoint 1500 \
    --stage-2-checkpoint 1500 \
    --stage-3-checkpoint 1500 \
    --headless \
    --total-timesteps 500000
```

### Hierarchical IL PPO

RL learns optimal sequencing of trained SADP_G stages:

```bash
# Test with dummy IL policy
python Env_RL/train_hierarchical_il_ppo.py --use-dummy-il

# With actual SADP_G checkpoints
python Env_RL/train_hierarchical_il_ppo.py \
    --training-data-num 100 \
    --stage-1-checkpoint 1500 \
    --stage-2-checkpoint 1500 \
    --stage-3-checkpoint 1500 \
    --total-timesteps 50000
```

### Monitoring Training

```bash
tensorboard --logdir logs/
```

---

## Architecture Details

### 1. Basic PPO (`fold_tops_gym_env.py`)

Standard end-to-end RL with Gymnasium interface.

#### Observation Space

| Component | Shape | Description |
|-----------|-------|-------------|
| `garment_pcd` | (2048, 3) | Downsampled garment point cloud |
| `joint_positions` | (60,) | All joint positions (both arms + hands) |
| `ee_poses` | (14,) | End-effector poses (pos + quat for each hand) |
| `gam_keypoints` | (6, 3) | GAM-detected manipulation keypoints |

#### Action Space

| Index | Description | Range |
|-------|-------------|-------|
| 0-2 | Left hand delta position (x, y, z) | [-1, 1] × action_scale |
| 3-5 | Right hand delta position (x, y, z) | [-1, 1] × action_scale |
| 6 | Left gripper command (>0.5 = close) | [0, 1] |
| 7 | Right gripper command (>0.5 = close) | [0, 1] |

---

### 2. Hierarchical PPO (`hierarchical_fold_env.py`)

High-level RL selects hand-coded manipulation primitives.

```
┌─────────────────────────────────────────────────────────┐
│                 Hierarchical PPO                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Observation → RL Policy → Primitive Selection          │
│                      │                                   │
│                      ▼                                   │
│   ┌─────────────────────────────────────────────┐       │
│   │        IK-Based Manipulation Primitives      │       │
│   │  • FOLD_LEFT_SLEEVE   (grasp→lift→fold)     │       │
│   │  • FOLD_RIGHT_SLEEVE  (grasp→lift→fold)     │       │
│   │  • FOLD_BOTTOM        (bimanual fold)        │       │
│   │  • OPEN_HANDS / MOVE_TO_HOME / DONE          │       │
│   └─────────────────────────────────────────────┘       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### Action Space (Discrete)

| ID | Primitive | Description |
|----|-----------|-------------|
| 0 | `FOLD_LEFT_SLEEVE` | IK trajectory to fold left sleeve |
| 1 | `FOLD_RIGHT_SLEEVE` | IK trajectory to fold right sleeve |
| 2 | `FOLD_BOTTOM` | Bimanual IK to fold bottom up |
| 3 | `OPEN_HANDS` | Open both grippers |
| 4 | `MOVE_TO_HOME` | Return to home position |
| 5 | `DONE` | Signal task completion |

---

### 3. Hierarchical IL PPO (`hierarchical_il_fold_env.py`)

**Key improvement**: Uses trained SADP_G diffusion policies instead of hand-coded IK.

```
┌─────────────────────────────────────────────────────────┐
│              Hierarchical IL PPO (NEW!)                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Observation → RL Policy → SADP_G Stage Selection       │
│                      │                                   │
│                      ▼                                   │
│   ┌─────────────────────────────────────────────┐       │
│   │        Trained SADP_G Diffusion Models       │       │
│   │  • STAGE_1_LEFT_SLEEVE  → SADP_G Stage 1    │       │
│   │  • STAGE_2_RIGHT_SLEEVE → SADP_G Stage 2    │       │
│   │  • STAGE_3_BOTTOM_FOLD  → SADP_G Stage 3    │       │
│   │  • Utility: OPEN_HANDS / MOVE_HOME / DONE    │       │
│   └─────────────────────────────────────────────┘       │
│                                                          │
│   Advantages:                                            │
│   • Much faster learning (sequence only)                 │
│   • Leverages pre-trained SADP_G models                  │
│   • More robust manipulation execution                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### Key Difference from Regular Hierarchical PPO

| Aspect | Hierarchical PPO | Hierarchical IL PPO |
|--------|------------------|---------------------|
| Primitives | Hard-coded IK trajectories | Trained SADP_G models |
| Manipulation | Less robust | More robust (from demos) |
| What RL learns | Sequence + timing | Sequence + timing |
| Low-level control | Scripted IK | Diffusion policy |

#### Observation Space

| Component | Shape | Description |
|-----------|-------|-------------|
| `garment_pcd` | (2048, 3) | Garment point cloud |
| `gam_keypoints` | (6, 3) | GAM manipulation points |
| `primitive_mask` | (6,) | Available primitives (1=available) |
| `executed_sequence` | (6,) | One-hot of executed stages |

---

### 4. Multi-Stage Residual PPO (`multi_stage_residual_env.py`) ⭐ RECOMMENDED

The most sophisticated approach: single RL policy learns **small corrections** to SADP_G actions AND **when to transition** between stages.

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Stage Residual PPO                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Observation:                                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ garment_pcd │ │ ee_poses    │ │ gam_points  │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ il_action   │ │ stage_id    │ │ stage_prog  │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│                         │                                   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────┐           │
│  │           Single RL Policy (PPO)            │           │
│  └─────────────────────────────────────────────┘           │
│                         │                                   │
│          ┌──────────────┴──────────────┐                   │
│          ▼                              ▼                   │
│  ┌───────────────┐              ┌───────────────┐          │
│  │ Residuals(60D)│              │ Stage Advance │          │
│  └───────────────┘              └───────────────┘          │
│          │                              │                   │
│          ▼                              ▼                   │
│  final_action = IL_action + residual   If > 0.5: advance   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Key Features

1. **Single RL Policy for All Stages**: One network handles all 3 folding stages
2. **Learned Stage Transitions**: RL decides WHEN to transition (not hard-coded)
3. **Stage-Conditioned Observations**: Policy receives stage indicator for stage-aware behavior
4. **Bounded Residuals**: IL provides strong baseline, RL learns small corrections
5. **Joint-Space Actions**: 60D actions match SADP_G output directly

#### Action Space (61D Continuous)

| Index | Description | Scale |
|-------|-------------|-------|
| 0-5 | Left arm joint residuals | ±0.05 rad |
| 6-29 | Left hand joint residuals | ±0.01 rad |
| 30-35 | Right arm joint residuals | ±0.05 rad |
| 36-59 | Right hand joint residuals | ±0.01 rad |
| 60 | Stage advance signal | >0.5 = advance |

#### Quick Wins: Surgical Residual Application

Since IL already performs well, RL makes **surgical interventions**:

| Feature | Description | Why It Helps |
|---------|-------------|--------------|
| `arm_only_residuals=True` | Only arm joints get residuals, hands are 100% IL | Preserves grasp configuration |
| `residual_apply_interval=5` | Residuals only applied every N steps | Reduces noise, lets IL shine |
| `disable_residual_during_manipulation=True` | Pure IL when gripper closed | Protects delicate manipulation |

#### Stage-Specific Observations

| Component | Shape | Description |
|-----------|-------|-------------|
| `current_stage` | (4,) | One-hot: [stage1, stage2, stage3, completed] |
| `stage_progress` | (1,) | Normalized progress within current stage |
| `stages_completed` | (3,) | Binary mask of completed stages |
| `il_action` | (60,) | Joint action from current stage's SADP_G |

---

### 5. Residual RL (`residual_fold_env.py`)

Single-stage residual learning (simpler than multi-stage).

```
┌─────────────────────────────────────────────────────────┐
│                     Residual RL                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Observation → IL Policy (frozen) → Base Action         │
│                         +                                │
│   Observation → RL Policy (trainable) → Residual         │
│                         =                                │
│                    Final Action                          │
│                                                          │
│   final_action = IL_action + bounded_residual            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### Action Space (8D Continuous)

| Index | Description | Range |
|-------|-------------|-------|
| 0-2 | Left EE residual (x, y, z) | ±residual_scale |
| 3-5 | Right EE residual (x, y, z) | ±residual_scale |
| 6 | Left gripper delta | ±residual_scale |
| 7 | Right gripper delta | ±residual_scale |

---

## Reward Functions

### Basic PPO Reward

```python
reward = fold_progress × 1.0      # Reduction in XY bounding box area
       + compactness × 0.5        # Reduction in total volume
       - height_penalty × 0.3     # Penalize height variance
       - action_penalty × 0.01    # Smooth actions
       + success_bonus × 10.0     # Task completion bonus
```

### Multi-Stage Residual Reward

```python
reward = fold_progress × 1.0           # Overall folding progress
       + compactness × 0.5             # Volume reduction
       - height_penalty × 0.3          # Keep garment flat
       - residual_penalty × 0.02       # Encourage small corrections
       - smoothness_penalty × 0.05     # Penalize jerky movements
       + stage_advance_bonus × 5.0     # Reward good stage transitions
       - premature_advance × 2.0       # Penalize early transitions
       + stage_completion × 3.0        # Per-stage completion bonus
       + task_success_bonus × 20.0     # Final completion reward
```

### Hierarchical IL PPO Reward

```python
reward = +5.0    # For each successful IL stage
       + +15.0   # Bonus for completing all three stages
       + quality × 3.0  # Up to +3 for fold quality
       - 1.0     # For failed stage
       - 0.5     # For re-selecting executed stage
       - 2.0     # For premature DONE
```

---

## Success Criteria

An episode is successful when:

- Bounding box X dimension < 50% of initial
- Bounding box Y dimension < 70% of initial
- Height variance < 0.02

---

## Configuration

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--learning-rate` | 3e-4 | PPO learning rate |
| `--n-steps` | 256-512 | Steps per PPO update |
| `--batch-size` | 64 | Mini-batch size |
| `--n-epochs` | 10 | Epochs per update |
| `--gamma` | 0.99 | Discount factor |
| `--gae-lambda` | 0.95 | GAE lambda |
| `--clip-range` | 0.2 | PPO clip range |
| `--ent-coef` | 0.01-0.1 | Entropy coefficient |

### Environment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_episode_steps` | 300-400 | Maximum steps per episode |
| `action_scale` | 0.05 | Scaling factor for actions (meters) |
| `point_cloud_size` | 2048 | Number of points in observation |
| `use_gam_features` | True | Include GAM keypoints in obs |

### Multi-Stage Specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `arm_residual_scale` | 0.05 | Max arm joint residual |
| `hand_residual_scale` | 0.01 | Max hand joint residual |
| `stage_advance_threshold` | 0.5 | Threshold to trigger advance |
| `min_steps_before_advance` | 10 | Min steps before can advance |
| `use_fixed_initial_state` | True | Match SADP_G training distribution |

---

## File Structure

```
Env_RL/
├── __init__.py                    # Module initialization
├── README.md                      # This documentation
│
├── # ═══════════════════════════════════════════════════════
├── # BASIC PPO (End-to-End RL)
├── # ═══════════════════════════════════════════════════════
├── fold_tops_gym_env.py           # Basic Gymnasium environment wrapper
├── train_ppo.py                   # Basic PPO training script
├── eval_ppo.py                    # Basic evaluation script
│
├── # ═══════════════════════════════════════════════════════
├── # HIERARCHICAL PPO (IK Primitives)
├── # ═══════════════════════════════════════════════════════
├── primitives.py                  # Hand-coded IK manipulation primitives
├── hierarchical_fold_env.py       # Hierarchical environment
├── train_hierarchical_ppo.py      # Hierarchical PPO training
├── eval_hierarchical.py           # Hierarchical evaluation
│
├── # ═══════════════════════════════════════════════════════
├── # HIERARCHICAL IL PPO (SADP_G Primitives) ⭐ NEW
├── # ═══════════════════════════════════════════════════════
├── il_primitives.py               # SADP_G-based manipulation primitives
├── hierarchical_il_fold_env.py    # Hierarchical IL environment
├── train_hierarchical_il_ppo.py   # Hierarchical IL PPO training
│
├── # ═══════════════════════════════════════════════════════
├── # RESIDUAL RL (IL + RL Corrections)
├── # ═══════════════════════════════════════════════════════
├── il_policy_wrapper.py           # Wrapper for IL policies (SADP, DP3)
├── residual_fold_env.py           # Single-stage residual environment
├── train_residual_ppo.py          # Residual PPO training
│
├── # ═══════════════════════════════════════════════════════
├── # MULTI-STAGE RESIDUAL RL (RECOMMENDED) ⭐
├── # ═══════════════════════════════════════════════════════
├── multi_stage_sadpg_wrapper.py   # Wrapper for 3 SADP_G stage models
├── multi_stage_residual_env.py    # Multi-stage environment with learned transitions
├── train_multi_stage_ppo.py       # Multi-stage PPO training
└── eval_multi_stage.py            # Multi-stage evaluation
```

---

## Evaluation

### Basic PPO

```bash
python Env_RL/eval_ppo.py \
    --model-path checkpoints/ppo_fold_tops/best/best_model.zip \
    --n-episodes 10

# With video recording
python Env_RL/eval_ppo.py \
    --model-path checkpoints/ppo_fold_tops/best/best_model.zip \
    --record-video --video-dir eval_videos
```

### Hierarchical PPO

```bash
python Env_RL/eval_hierarchical.py \
    --model-path checkpoints/hierarchical_ppo/best/best_model.zip \
    --n-episodes 20 \
    --verbose
```

### Multi-Stage Residual PPO

```bash
python Env_RL/eval_multi_stage.py \
    --model-path checkpoints/multi_stage_ppo/best/best_model.zip \
    --n-episodes 20

# Compare with IL-only baseline
python Env_RL/eval_multi_stage.py \
    --model-path checkpoints/multi_stage_ppo/best/best_model.zip \
    --compare-il
```

---

## Tips for Training

### General Tips

1. **Start with GUI**: Debug with `--headless=False` first to visualize behavior
2. **Use GAM features**: They provide semantic keypoints that help the policy
3. **Checkpoint frequently**: Training can be interrupted; use `--save-freq`
4. **Monitor TensorBoard**: Watch for reward curves and episode lengths

### Multi-Stage Specific Tips

1. **Match Initial State**: Use `--use-fixed-initial-state` to match SADP_G training distribution
2. **Start with dummy IL**: Test pipeline with `--use-dummy-il` before loading real models
3. **Surgical Residuals**: Enable `arm_only_residuals` to let IL control grasping
4. **Tune Stage Timing**: Adjust `min_steps_before_advance` based on your task

### Hyperparameter Recommendations

| Approach | n_steps | batch_size | ent_coef | Notes |
|----------|---------|------------|----------|-------|
| Basic PPO | 256 | 64 | 0.01 | Standard continuous control |
| Hierarchical | 64 | 32 | 0.1 | Higher entropy for discrete |
| Hierarchical IL | 64 | 32 | 0.1 | Faster learning (sequence only) |
| Multi-Stage | 512 | 128 | 0.01 | Larger batches for joint-space |

---

## Known Limitations

1. **Single environment**: Particle cloth doesn't support vectorized envs yet
2. **Training speed**: ~1-5 FPS due to physics simulation overhead
3. **Memory usage**: Point cloud processing can be memory intensive
4. **SADP_G dependency**: Multi-stage and hierarchical IL require trained SADP_G models

---

## Extending to Other Tasks

To create a new RL environment for another task:

1. Copy the appropriate base environment as a template
2. Modify `_lazy_init()` to set up your scene (garment type, robot config)
3. Implement task-specific `_compute_reward()` and `_check_success()`
4. Adjust observation/action spaces as needed
5. For IL-based approaches, ensure SADP_G models are trained for your task

### Example: Creating a Hang Tops Environment

```python
# 1. Copy multi_stage_residual_env.py
# 2. Change garment path in _lazy_init()
garment_usd = "Assets/Garment/Tops/.../your_tops.usd"

# 3. Update reward function for hanging task
def _compute_reward(self, obs, residual, ...):
    # Reward garment being lifted and positioned on hanger
    height_reward = garment_height * self.reward_weights["height"]
    alignment_reward = ...
    return reward

# 4. Update success criteria
def _check_final_success(self, obs):
    # Garment is on hanger and released
    return garment_on_hanger and hands_open
```

---

## References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [DexGarmentLab Paper](https://github.com/DexGarmentLab/DexGarmentLab)
- [SADP: Semantic Affordance-Driven Diffusion Policy](Model_HALO/SADP/)
