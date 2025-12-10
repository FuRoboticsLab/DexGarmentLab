#!/usr/bin/env python3
"""
Residual PPO Training Script for DexGarmentLab.

This script trains a RESIDUAL policy that learns corrections on top of
a frozen IL policy (SADP, DP3, etc.).

Architecture:
    Final Action = IL_Action (frozen) + RL_Residual (trained)

The RL policy only learns small corrections, which is much easier than
learning the full action from scratch.

Usage:
    # Test with dummy IL policy (no checkpoint needed)
    python Env_RL/train_residual_ppo.py --il-policy dummy
    
    # With actual SADP policy
    python Env_RL/train_residual_ppo.py --il-policy SADP --task-name Fold_Tops --checkpoint 1000

Key Benefits:
    - IL policy provides strong baseline behavior
    - RL only needs to learn corrections (bounded residuals)
    - Much faster convergence than learning from scratch
    - IL policy is frozen - no risk of catastrophic forgetting
"""

from isaacsim import SimulationApp
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train Residual PPO")
    
    # Environment settings
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--max-episode-steps", type=int, default=300)
    
    # IL Policy settings
    parser.add_argument(
        "--il-policy",
        type=str,
        default="dummy",
        choices=["dummy", "SADP", "SADP_G", "DP3"],
        help="IL policy type to use"
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="Fold_Tops",
        help="Task name for IL checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=1000,
        help="IL policy checkpoint number"
    )
    parser.add_argument(
        "--data-num",
        type=int,
        default=100,
        help="IL policy data configuration"
    )
    
    # Residual settings
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.1,
        help="Max magnitude of residual actions"
    )
    
    # Training settings
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--save-freq", type=int, default=10000)
    parser.add_argument("--eval-freq", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    
    # Logging
    parser.add_argument("--log-dir", type=str, default="./logs/residual_ppo")
    parser.add_argument("--save-dir", type=str, default="./checkpoints/residual_ppo")
    parser.add_argument("--load-path", type=str, default=None)
    
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

import os
import sys
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
        CallbackList,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.logger import configure
except ImportError:
    print("ERROR: stable-baselines3 not installed!")
    simulation_app.close()
    sys.exit(1)

from Env_RL.residual_fold_env import ResidualFoldEnv


class ResidualPolicyNetwork:
    """
    Custom policy network for residual RL.
    
    Key design choices:
    - Takes IL action as input (helps RL understand what IL would do)
    - Outputs small bounded residuals
    - Lighter architecture since task is simpler
    """
    
    @staticmethod
    def get_policy_kwargs():
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        import torch
        import torch.nn as nn
        
        class ResidualFeaturesExtractor(BaseFeaturesExtractor):
            """Feature extractor for residual policy."""
            
            def __init__(self, observation_space, features_dim: int = 128):
                super().__init__(observation_space, features_dim)
                
                # Point cloud encoder (lightweight)
                self.pcd_encoder = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(2048 * 3, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                )
                
                # Joint position encoder
                self.joint_encoder = nn.Sequential(
                    nn.Linear(60, 32),
                    nn.ReLU(),
                )
                
                # EE pose encoder
                self.ee_encoder = nn.Sequential(
                    nn.Linear(14, 16),
                    nn.ReLU(),
                )
                
                # IL action encoder (IMPORTANT: helps RL see what IL would do)
                self.il_action_encoder = nn.Sequential(
                    nn.Linear(8, 16),
                    nn.ReLU(),
                )
                
                # GAM keypoints encoder
                self.gam_encoder = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(6 * 3, 16),
                    nn.ReLU(),
                )
                
                # Combiner: 64 + 32 + 16 + 16 + 16 = 144
                self.combiner = nn.Sequential(
                    nn.Linear(144, features_dim),
                    nn.ReLU(),
                )
                
            def forward(self, observations):
                pcd_features = self.pcd_encoder(observations["garment_pcd"])
                joint_features = self.joint_encoder(observations["joint_positions"])
                ee_features = self.ee_encoder(observations["ee_poses"])
                il_action_features = self.il_action_encoder(observations["il_action"])
                gam_features = self.gam_encoder(observations["gam_keypoints"])
                
                combined = torch.cat([
                    pcd_features,
                    joint_features,
                    ee_features,
                    il_action_features,
                    gam_features,
                ], dim=1)
                
                return self.combiner(combined)
        
        return {
            "features_extractor_class": ResidualFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": 128},
            "net_arch": dict(pi=[64, 32], vf=[64, 32]),
        }


def main():
    print("=" * 60)
    print("Residual PPO Training for Garment Folding")
    print("=" * 60)
    print(f"IL Policy: {args.il_policy}")
    if args.il_policy != "dummy":
        print(f"  Task: {args.task_name}")
        print(f"  Checkpoint: {args.checkpoint}")
    print(f"Residual Scale: ±{args.residual_scale}")
    print(f"Headless: {args.headless}")
    print(f"Total Timesteps: {args.total_timesteps}")
    print("=" * 60)
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"residual_ppo_{args.il_policy}_{timestamp}"
    
    # Create environment
    print("\n[INFO] Creating Residual RL environment...")
    env = ResidualFoldEnv(
        il_policy_type=args.il_policy,
        task_name=args.task_name,
        checkpoint_num=args.checkpoint,
        data_num=args.data_num,
        render_mode="human" if not args.headless else "rgb_array",
        max_episode_steps=args.max_episode_steps,
        residual_scale=args.residual_scale,
    )
    env = Monitor(env, filename=os.path.join(args.log_dir, run_name))
    
    # Get policy kwargs
    policy_kwargs = ResidualPolicyNetwork.get_policy_kwargs()
    
    # Create model
    if args.load_path and os.path.exists(args.load_path):
        print(f"\n[INFO] Loading model from {args.load_path}")
        model = PPO.load(args.load_path, env=env)
    else:
        print("\n[INFO] Creating new Residual PPO model...")
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=args.seed,
            verbose=1,
            tensorboard_log=args.log_dir,
            policy_kwargs=policy_kwargs,
            device="cuda",
        )
    
    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.save_dir,
        name_prefix=run_name,
    )
    
    eval_cb = EvalCallback(
        env,
        best_model_save_path=os.path.join(args.save_dir, "best"),
        log_path=args.log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=3,
        deterministic=True,
    )
    
    callbacks = CallbackList([checkpoint_cb, eval_cb])
    
    # Logger
    logger = configure(args.log_dir, ["stdout", "tensorboard"])
    model.set_logger(logger)
    
    # Print info
    print("\n[INFO] Model Summary:")
    print(f"  Policy: MultiInputPolicy with ResidualFeaturesExtractor")
    print(f"  Action Space: Box({-args.residual_scale}, {args.residual_scale}, (8,))")
    print(f"  The RL policy outputs RESIDUALS added to IL actions")
    
    # Train
    print("\n" + "=" * 60)
    print("Starting Residual RL Training...")
    print(f"  Final Action = IL_Action + RL_Residual")
    print(f"  Residual bounded to ±{args.residual_scale}")
    print(f"Monitor: tensorboard --logdir={args.log_dir}")
    print("=" * 60 + "\n")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=10,
            tb_log_name=run_name,
            reset_num_timesteps=args.load_path is None,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted.")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        final_path = os.path.join(args.save_dir, f"{run_name}_final")
        model.save(final_path)
        print(f"\n[INFO] Model saved to {final_path}")
    
    env.close()
    simulation_app.close()
    print("\n[INFO] Training complete!")


if __name__ == "__main__":
    main()



