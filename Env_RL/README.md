# DexGarmentLab RL Module

This module provides reinforcement learning capabilities for garment manipulation tasks using PPO (Proximal Policy Optimization).

## Overview

The RL module wraps existing DexGarmentLab environments with a Gymnasium interface, enabling training with standard RL libraries like Stable-Baselines3.

## Installation

Install the additional RL dependencies:

```bash
pip install gymnasium stable-baselines3[extra] tensorboard opencv-python
```

## Quick Start

### Training

Train a PPO agent on the Fold Tops task:

```bash
# With GUI (for debugging)
python Env_RL/train_ppo.py --total-timesteps 500000

# Headless mode (faster training)
python Env_RL/train_ppo.py --headless --total-timesteps 1000000

# With custom garment
python Env_RL/train_ppo.py --garment-usd "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Jacket032/TCLC_Jacket032_obj.usd"
```

### Evaluation

Evaluate a trained model:

```bash
python Env_RL/eval_ppo.py --model-path checkpoints/ppo_fold_tops/best/best_model.zip --n-episodes 10

# With video recording
python Env_RL/eval_ppo.py --model-path checkpoints/ppo_fold_tops/best/best_model.zip --record-video --video-dir eval_videos
```

### Monitoring Training

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir logs/ppo_fold_tops
```

## Architecture

### Observation Space

| Component | Shape | Description |
|-----------|-------|-------------|
| `garment_pcd` | (2048, 3) | Downsampled garment point cloud |
| `joint_positions` | (60,) | All joint positions (both arms + hands) |
| `ee_poses` | (14,) | End-effector poses (pos + quat for each hand) |
| `gam_keypoints` | (6, 3) | GAM-detected manipulation keypoints (optional) |

### Action Space

| Index | Description | Range |
|-------|-------------|-------|
| 0-2 | Left hand delta position (x, y, z) | [-1, 1] × action_scale |
| 3-5 | Right hand delta position (x, y, z) | [-1, 1] × action_scale |
| 6 | Left gripper command (>0.5 = close) | [0, 1] |
| 7 | Right gripper command (>0.5 = close) | [0, 1] |

### Reward Function

The reward function encourages successful garment folding:

```
reward = fold_progress × 1.0      # Reduction in XY bounding box area
       + compactness × 0.5        # Reduction in total volume
       - height_penalty × 0.3     # Penalize height variance
       - action_penalty × 0.01    # Smooth actions
       + success_bonus × 10.0     # Task completion bonus
```

### Success Criteria

An episode is successful when:
- Bounding box X dimension < 50% of initial
- Bounding box Y dimension < 70% of initial
- Height variance < 0.02

## Configuration

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--learning-rate` | 3e-4 | PPO learning rate |
| `--n-steps` | 256 | Steps per PPO update |
| `--batch-size` | 64 | Mini-batch size |
| `--n-epochs` | 10 | Epochs per update |
| `--gamma` | 0.99 | Discount factor |
| `--gae-lambda` | 0.95 | GAE lambda |
| `--clip-range` | 0.2 | PPO clip range |
| `--ent-coef` | 0.01 | Entropy coefficient |

### Environment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_episode_steps` | 300 | Maximum steps per episode |
| `action_scale` | 0.05 | Scaling factor for actions (meters) |
| `point_cloud_size` | 2048 | Number of points in observation |
| `use_gam_features` | True | Include GAM keypoints in obs |

## File Structure

```
Env_RL/
├── __init__.py              # Module initialization
├── fold_tops_gym_env.py     # Gymnasium environment wrapper
├── train_ppo.py             # PPO training script
├── eval_ppo.py              # Evaluation script
└── README.md                # This file
```

## Tips for Training

1. **Start with GUI**: Debug with `--headless=False` first to visualize behavior
2. **Adjust action_scale**: Smaller values = finer control but slower movement
3. **Tune rewards**: Adjust `reward_weights` in the environment for your task
4. **Use GAM features**: They provide semantic keypoints that help the policy
5. **Checkpoint frequently**: Training can be interrupted; use `--save-freq`

## Known Limitations

1. **Single environment**: Particle cloth doesn't support vectorized envs yet
2. **Training speed**: ~1-5 FPS due to physics simulation overhead
3. **Memory usage**: Point cloud processing can be memory intensive

## Extending to Other Tasks

To create a new RL environment for another task:

1. Copy `fold_tops_gym_env.py` as a template
2. Modify `_lazy_init()` to set up your scene
3. Implement task-specific `_compute_reward()` and `_check_success()`
4. Adjust observation/action spaces as needed

## References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

