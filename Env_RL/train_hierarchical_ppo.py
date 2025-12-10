#!/usr/bin/env python3
"""
Hierarchical PPO Training Script for DexGarmentLab.

This script trains a HIGH-LEVEL policy to select manipulation primitives
(fold_left, fold_right, fold_bottom, etc.) while LOW-LEVEL execution
is handled by pre-defined controllers from DexGarmentLab.

Key Advantages over Flat RL:
- Action space: 6 discrete choices vs 60+ continuous DOF
- Much faster learning (primitives handle complex motor control)
- More interpretable policy (sequence of primitives)
- Reusable primitives across tasks

Usage:
    python Env_RL/train_hierarchical_ppo.py --total-timesteps 50000
    
    # The hierarchical approach needs far fewer timesteps since
    # action space is much simpler!
"""

# SimulationApp must be created FIRST
from isaacsim import SimulationApp
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hierarchical PPO on FoldTops task")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=50000,  # Much less needed for hierarchical!
        help="Total timesteps for training"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=5000,
        help="Save checkpoint every N timesteps"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Evaluate every N timesteps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=64,  # Smaller buffer for discrete actions
        help="Steps per update"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Epochs per update"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/hierarchical_ppo",
        help="Log directory"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints/hierarchical_ppo",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--load-path",
        type=str,
        default=None,
        help="Path to load pretrained model"
    )
    
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
    print("Install with: pip install stable-baselines3[extra]")
    simulation_app.close()
    sys.exit(1)

from Env_RL.hierarchical_fold_env import HierarchicalFoldEnv
from Env_RL.primitives import PrimitiveID


class HierarchicalPolicyNetwork:
    """
    Custom policy network for hierarchical environment.
    
    Handles Dict observations with:
    - Point cloud (2048, 3) - simplified encoder
    - GAM keypoints (6, 3) - semantic manipulation points
    - Primitive mask (6,) - which primitives are available
    - Executed sequence (6,) - which primitives have been done
    """
    
    @staticmethod
    def get_policy_kwargs():
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        import torch
        import torch.nn as nn
        
        class HierarchicalFeaturesExtractor(BaseFeaturesExtractor):
            """Feature extractor for hierarchical policy."""
            
            def __init__(self, observation_space, features_dim: int = 128):
                super().__init__(observation_space, features_dim)
                
                # Point cloud encoder (lightweight for faster training)
                self.pcd_encoder = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(2048 * 3, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                )
                
                # GAM keypoints encoder (most important for primitive selection!)
                self.gam_encoder = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(6 * 3, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                    nn.ReLU(),
                )
                
                # Primitive state encoder
                self.state_encoder = nn.Sequential(
                    nn.Linear(12, 16),  # mask (6) + executed (6)
                    nn.ReLU(),
                )
                
                # Combiner
                # 64 (pcd) + 32 (gam) + 16 (state) = 112
                self.combiner = nn.Sequential(
                    nn.Linear(112, features_dim),
                    nn.ReLU(),
                )
                
            def forward(self, observations):
                pcd_features = self.pcd_encoder(observations["garment_pcd"])
                gam_features = self.gam_encoder(observations["gam_keypoints"])
                
                # Concatenate mask and executed sequence
                state = torch.cat([
                    observations["primitive_mask"],
                    observations["executed_sequence"]
                ], dim=1)
                state_features = self.state_encoder(state)
                
                # Combine all features
                combined = torch.cat([
                    pcd_features, gam_features, state_features
                ], dim=1)
                
                return self.combiner(combined)
        
        return {
            "features_extractor_class": HierarchicalFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": 128},
            "net_arch": dict(pi=[64, 32], vf=[64, 32]),  # Smaller network
        }


def main():
    print("=" * 60)
    print("Hierarchical PPO Training for Garment Folding")
    print("=" * 60)
    print(f"Action space: 6 discrete primitives")
    print(f"  0: FOLD_LEFT_SLEEVE")
    print(f"  1: FOLD_RIGHT_SLEEVE")
    print(f"  2: FOLD_BOTTOM")
    print(f"  3: OPEN_HANDS")
    print(f"  4: MOVE_TO_HOME")
    print(f"  5: DONE")
    print("=" * 60)
    print(f"Headless: {args.headless}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"hierarchical_ppo_{timestamp}"
    
    # Create environment
    print("\n[INFO] Creating hierarchical environment...")
    env = HierarchicalFoldEnv(
        config={},
        render_mode="human" if not args.headless else "rgb_array",
        max_primitives=10,
        point_cloud_size=2048,
    )
    env = Monitor(env, filename=os.path.join(args.log_dir, run_name))
    
    # Get policy kwargs
    policy_kwargs = HierarchicalPolicyNetwork.get_policy_kwargs()
    
    # Create PPO model
    if args.load_path and os.path.exists(args.load_path):
        print(f"\n[INFO] Loading model from {args.load_path}")
        model = PPO.load(args.load_path, env=env)
    else:
        print("\n[INFO] Creating new PPO model...")
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
            ent_coef=0.1,  # Higher entropy for exploration of primitives
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=args.seed,
            verbose=1,
            tensorboard_log=args.log_dir,
            policy_kwargs=policy_kwargs,
            device="cuda",
        )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.save_dir,
        name_prefix=run_name,
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(args.save_dir, "best"),
        log_path=args.log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=3,
        deterministic=True,
    )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # Configure logger
    logger = configure(args.log_dir, ["stdout", "tensorboard"])
    model.set_logger(logger)
    
    # Print model info
    print("\n[INFO] Model Summary:")
    print(f"  Policy: {model.policy}")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space keys: {list(env.observation_space.spaces.keys())}")
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print(f"Monitor with: tensorboard --logdir={args.log_dir}")
    print("=" * 60 + "\n")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=5,
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
        # Save final model
        final_path = os.path.join(args.save_dir, f"{run_name}_final")
        model.save(final_path)
        print(f"\n[INFO] Model saved to {final_path}")
    
    env.close()
    simulation_app.close()
    print("\n[INFO] Training complete!")


if __name__ == "__main__":
    main()



