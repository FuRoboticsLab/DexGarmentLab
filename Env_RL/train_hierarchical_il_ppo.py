#!/usr/bin/env python3
"""
Hierarchical PPO Training Script with IL (SADP_G) Primitives.

This script trains a HIGH-LEVEL policy to select which SADP_G stage to execute,
while the LOW-LEVEL execution is handled by trained SADP_G diffusion policies.

Key Difference from train_hierarchical_ppo.py:
- OLD: Primitives use hard-coded IK trajectories
- NEW: Primitives use trained SADP_G models for each stage

Architecture:
    High-Level RL: Selects stage (0=Left, 1=Right, 2=Bottom, 3=Open, 4=Home, 5=Done)
    Low-Level IL: SADP_G models execute the selected stage

Key Advantages:
- Much faster learning (only learns sequence, not motor control)
- Leverages pre-trained SADP_G models that already work well
- More interpretable (sequence of stages)
- Action space: 6 discrete choices vs 60+ continuous DOF

Usage:
    # Test with dummy IL policy
    python Env_RL/train_hierarchical_il_ppo.py --use-dummy-il

    # With actual SADP_G checkpoints
    python Env_RL/train_hierarchical_il_ppo.py \\
        --training-data-num 100 \\
        --stage-1-checkpoint 1500 \\
        --stage-2-checkpoint 1500 \\
        --stage-3-checkpoint 1500
"""

from isaacsim import SimulationApp
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hierarchical IL PPO")
    
    # Environment settings
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--max-primitives", type=int, default=10,
                        help="Maximum primitives per episode")
    
    # SADP_G settings (THE KEY ADDITION!)
    parser.add_argument("--use-dummy-il", action="store_true",
                        help="Use dummy IL policy for testing")
    parser.add_argument("--training-data-num", type=int, default=100,
                        help="Training data config for SADP_G")
    parser.add_argument("--stage-1-checkpoint", type=int, default=1500,
                        help="Checkpoint for stage 1 SADP_G")
    parser.add_argument("--stage-2-checkpoint", type=int, default=1500,
                        help="Checkpoint for stage 2 SADP_G")
    parser.add_argument("--stage-3-checkpoint", type=int, default=1500,
                        help="Checkpoint for stage 3 SADP_G")
    
    # Training settings
    parser.add_argument("--total-timesteps", type=int, default=50000,
                        help="Total timesteps (less needed for hierarchical)")
    parser.add_argument("--save-freq", type=int, default=5000,
                        help="Save checkpoint every N timesteps")
    parser.add_argument("--eval-freq", type=int, default=10000,
                        help="Evaluate every N timesteps")
    parser.add_argument("--n-eval-episodes", type=int, default=3,
                        help="Number of evaluation episodes")
    
    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=64,
                        help="Steps per update (smaller for discrete)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.1,
                        help="Entropy coefficient (higher for discrete exploration)")
    parser.add_argument("--seed", type=int, default=42)
    
    # Logging
    parser.add_argument("--log-dir", type=str, default="./logs/hierarchical_il_ppo")
    parser.add_argument("--save-dir", type=str, default="./checkpoints/hierarchical_il_ppo")
    parser.add_argument("--load-path", type=str, default=None,
                        help="Path to load pretrained model")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name for logging")
    
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

from Env_RL.hierarchical_il_fold_env import HierarchicalILFoldEnv
from Env_RL.il_primitives import ILPrimitiveID


class HierarchicalILPolicyNetwork:
    """
    Custom policy network for hierarchical IL environment.
    
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
        
        class HierarchicalILFeaturesExtractor(BaseFeaturesExtractor):
            """Feature extractor for hierarchical IL policy."""
            
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
                
                # GAM keypoints encoder (important for stage selection!)
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
                
                # Combiner: 64 (pcd) + 32 (gam) + 16 (state) = 112
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
            "features_extractor_class": HierarchicalILFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": 128},
            "net_arch": dict(pi=[64, 32], vf=[64, 32]),
        }


def main():
    print("=" * 70)
    print("Hierarchical IL PPO Training with SADP_G Primitives")
    print("=" * 70)
    print(f"IL Policy: {'Dummy (testing)' if args.use_dummy_il else 'SADP_G'}")
    if not args.use_dummy_il:
        print(f"  Stage 1 Checkpoint: {args.stage_1_checkpoint}")
        print(f"  Stage 2 Checkpoint: {args.stage_2_checkpoint}")
        print(f"  Stage 3 Checkpoint: {args.stage_3_checkpoint}")
        print(f"  Training Data: {args.training_data_num}")
    print("-" * 70)
    print("Action space: 6 discrete primitives (IL-based)")
    print(f"  0: STAGE_1_LEFT_SLEEVE  (SADP_G Stage 1)")
    print(f"  1: STAGE_2_RIGHT_SLEEVE (SADP_G Stage 2)")
    print(f"  2: STAGE_3_BOTTOM_FOLD  (SADP_G Stage 3)")
    print(f"  3: OPEN_HANDS")
    print(f"  4: MOVE_TO_HOME")
    print(f"  5: DONE")
    print("-" * 70)
    print(f"Headless: {args.headless}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 70)
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create experiment name
    if args.exp_name:
        run_name = args.exp_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        il_suffix = "dummy" if args.use_dummy_il else "sadpg"
        run_name = f"hier_il_{il_suffix}_{timestamp}"
    
    # Create environment with SADP_G models
    print("\n[INFO] Creating hierarchical IL environment with SADP_G...")
    env = HierarchicalILFoldEnv(
        training_data_num=args.training_data_num,
        stage_1_checkpoint=args.stage_1_checkpoint,
        stage_2_checkpoint=args.stage_2_checkpoint,
        stage_3_checkpoint=args.stage_3_checkpoint,
        use_dummy_il=args.use_dummy_il,
        config={},
        render_mode="human" if not args.headless else "rgb_array",
        max_primitives=args.max_primitives,
        point_cloud_size=2048,
    )
    env = Monitor(env, filename=os.path.join(args.log_dir, run_name))
    
    # Get policy kwargs
    policy_kwargs = HierarchicalILPolicyNetwork.get_policy_kwargs()
    
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
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
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
        n_eval_episodes=args.n_eval_episodes,
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
    print("\n" + "=" * 70)
    print("Starting Hierarchical IL Training")
    print("-" * 70)
    print("Architecture:")
    print("  HIGH-LEVEL: RL learns which SADP_G stage to execute")
    print("  LOW-LEVEL:  SADP_G models handle actual manipulation")
    print("")
    print("  RL selects: [Stage1, Stage2, Stage3, OpenHands, Home, Done]")
    print("  SADP_G executes: Trained diffusion policy for each stage")
    print("-" * 70)
    print(f"Monitor with: tensorboard --logdir={args.log_dir}")
    print("=" * 70 + "\n")
    
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
