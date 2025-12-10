#!/usr/bin/env python3
"""
PPO Training Script for DexGarmentLab Fold Tops Task.

This script trains a PPO agent to fold garments using the existing
DexGarmentLab simulation environment.

Usage:
    python Env_RL/train_ppo.py --total-timesteps 500000 --save-freq 10000

Requirements:
    - stable-baselines3
    - tensorboard
    - gymnasium
"""

# IMPORTANT: SimulationApp must be created BEFORE any other Isaac Sim imports
from isaacsim import SimulationApp

# Parse args first to get headless mode
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on FoldTops task")
    parser.add_argument(
        "--headless", 
        action="store_true", 
        help="Run in headless mode (no GUI)"
    )
    parser.add_argument(
        "--total-timesteps", 
        type=int, 
        default=500000,
        help="Total timesteps for training"
    )
    parser.add_argument(
        "--save-freq", 
        type=int, 
        default=10000,
        help="Save checkpoint every N timesteps"
    )
    parser.add_argument(
        "--eval-freq", 
        type=int, 
        default=20000,
        help="Evaluate agent every N timesteps"
    )
    parser.add_argument(
        "--n-eval-episodes", 
        type=int, 
        default=5,
        help="Number of episodes for evaluation"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=3e-4,
        help="Learning rate for PPO"
    )
    parser.add_argument(
        "--n-steps", 
        type=int, 
        default=256,
        help="Number of steps per PPO update"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=64,
        help="Batch size for PPO"
    )
    parser.add_argument(
        "--n-epochs", 
        type=int, 
        default=10,
        help="Number of epochs per PPO update"
    )
    parser.add_argument(
        "--gamma", 
        type=float, 
        default=0.99,
        help="Discount factor"
    )
    parser.add_argument(
        "--gae-lambda", 
        type=float, 
        default=0.95,
        help="GAE lambda parameter"
    )
    parser.add_argument(
        "--clip-range", 
        type=float, 
        default=0.2,
        help="PPO clip range"
    )
    parser.add_argument(
        "--ent-coef", 
        type=float, 
        default=0.01,
        help="Entropy coefficient"
    )
    parser.add_argument(
        "--vf-coef", 
        type=float, 
        default=0.5,
        help="Value function coefficient"
    )
    parser.add_argument(
        "--max-grad-norm", 
        type=float, 
        default=0.5,
        help="Max gradient norm for clipping"
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
        default="./logs/ppo_fold_tops",
        help="Directory for tensorboard logs"
    )
    parser.add_argument(
        "--save-dir", 
        type=str, 
        default="./checkpoints/ppo_fold_tops",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--load-path", 
        type=str, 
        default=None,
        help="Path to load pretrained model"
    )
    parser.add_argument(
        "--garment-usd", 
        type=str, 
        default=None,
        help="Path to garment USD file"
    )
    parser.add_argument(
        "--use-gam", 
        action="store_true",
        default=True,
        help="Use GAM features in observations"
    )
    parser.add_argument(
        "--no-gam", 
        action="store_true",
        help="Disable GAM features"
    )
    
    return parser.parse_args()


# Create SimulationApp
args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

# Now import other modules
import os
import sys
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RL libraries
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
    print("=" * 60)
    print("ERROR: stable-baselines3 not installed!")
    print("Please install with: pip install stable-baselines3[extra]")
    print("=" * 60)
    simulation_app.close()
    sys.exit(1)

# Import custom environment
from Env_RL.fold_tops_gym_env import FoldTopsGymEnv


class CustomPolicyNetwork:
    """
    Custom policy network configuration for handling Dict observations.
    
    Uses separate feature extractors for:
    - Point cloud: PointNet-style MLP
    - Joint positions: MLP
    - EE poses: MLP
    - GAM keypoints: MLP
    """
    
    @staticmethod
    def get_policy_kwargs(use_gam: bool = True):
        """Get policy kwargs for MultiInputPolicy."""
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        import torch
        import torch.nn as nn
        
        class GarmentFeaturesExtractor(BaseFeaturesExtractor):
            """Custom feature extractor for garment manipulation."""
            
            def __init__(self, observation_space, features_dim: int = 256):
                super().__init__(observation_space, features_dim)
                
                # Point cloud encoder (simplified PointNet)
                self.pcd_encoder = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(2048 * 3, 512),
                    nn.ReLU(),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                )
                
                # Joint position encoder
                self.joint_encoder = nn.Sequential(
                    nn.Linear(60, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                )
                
                # EE pose encoder
                self.ee_encoder = nn.Sequential(
                    nn.Linear(14, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                )
                
                # GAM keypoints encoder (if used)
                self.use_gam = "gam_keypoints" in observation_space.spaces
                if self.use_gam:
                    self.gam_encoder = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(6 * 3, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16),
                        nn.ReLU(),
                    )
                    combined_dim = 128 + 32 + 16 + 16
                else:
                    combined_dim = 128 + 32 + 16
                
                # Final combination layer
                self.combiner = nn.Sequential(
                    nn.Linear(combined_dim, features_dim),
                    nn.ReLU(),
                )
                
            def forward(self, observations):
                pcd_features = self.pcd_encoder(observations["garment_pcd"])
                joint_features = self.joint_encoder(observations["joint_positions"])
                ee_features = self.ee_encoder(observations["ee_poses"])
                
                if self.use_gam:
                    gam_features = self.gam_encoder(observations["gam_keypoints"])
                    combined = torch.cat([
                        pcd_features, joint_features, ee_features, gam_features
                    ], dim=1)
                else:
                    combined = torch.cat([
                        pcd_features, joint_features, ee_features
                    ], dim=1)
                
                return self.combiner(combined)
        
        return {
            "features_extractor_class": GarmentFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": dict(pi=[128, 64], vf=[128, 64]),
            "activation_fn": nn.ReLU,
        }


def main():
    """Main training function."""
    
    print("=" * 60)
    print("PPO Training for DexGarmentLab Fold Tops Task")
    print("=" * 60)
    print(f"Headless mode: {args.headless}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_fold_tops_{timestamp}"
    
    # Environment configuration
    env_config = {
        "usd_path": args.garment_usd,
        "ground_material_usd": None,
    }
    
    use_gam = args.use_gam and not args.no_gam
    
    # Create training environment
    print("\n[INFO] Creating training environment...")
    env = FoldTopsGymEnv(
        config=env_config,
        render_mode="human" if not args.headless else "rgb_array",
        max_episode_steps=300,
        action_scale=0.05,
        use_gam_features=use_gam,
        point_cloud_size=2048,
    )
    env = Monitor(env, filename=os.path.join(args.log_dir, run_name))
    
    # Create evaluation environment (optional - uses same env for simplicity)
    # In practice, you might want a separate eval env
    eval_env = env
    
    # Get policy kwargs
    policy_kwargs = CustomPolicyNetwork.get_policy_kwargs(use_gam=use_gam)
    
    # Create or load PPO model
    if args.load_path and os.path.exists(args.load_path):
        print(f"\n[INFO] Loading pretrained model from {args.load_path}")
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
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            verbose=1,
            tensorboard_log=args.log_dir,
            policy_kwargs=policy_kwargs,
            device="cuda",
        )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.save_dir,
        name_prefix=run_name,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.save_dir, "best"),
        log_path=args.log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    
    callback_list = CallbackList([checkpoint_callback, eval_callback])
    
    # Configure logger
    new_logger = configure(args.log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)
    
    # Print model summary
    print("\n[INFO] Model Summary:")
    print(f"  Policy: {model.policy}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Device: {model.device}")
    
    # Start training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("Monitor with: tensorboard --logdir=" + args.log_dir)
    print("=" * 60 + "\n")
    
    try:
        print("About to call model.learn()")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback_list,
            log_interval=10,
            tb_log_name=run_name,
            reset_num_timesteps=args.load_path is None,
            progress_bar=True,
        )
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        print(f"Full error traceback:")
        import traceback
        traceback.print_exc()
        raise  # Re-raise to see the full error
    finally:
        # Save final model
        final_model_path = os.path.join(args.save_dir, f"{run_name}_final")
        model.save(final_model_path)
        print(f"\n[INFO] Final model saved to {final_model_path}")
    
    # Cleanup
    env.close()
    simulation_app.close()
    
    print("\n[INFO] Training complete!")


if __name__ == "__main__":
    main()



