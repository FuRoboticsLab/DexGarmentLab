#!/usr/bin/env python3
"""
SIMPLIFIED Multi-Stage Residual PPO Training Script.

This is a simplified version that removes complex mechanisms to focus on core learning.
"""

from isaacsim import SimulationApp
import argparse
import os
import sys
from datetime import datetime
import numpy as np

# Initialize Isaac Sim first
simulation_app = SimulationApp({"headless": False})

# Now import everything else
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Env_RL.multi_stage_residual_env_simple import MultiStageResidualEnvSimple


def parse_args():
    parser = argparse.ArgumentParser(description="Train SIMPLIFIED Multi-Stage Residual PPO")
    
    # Environment settings
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--max-episode-steps", type=int, default=400,
                        help="Maximum steps per episode")
    
    # SADP_G settings
    parser.add_argument("--use-dummy-il", action="store_true",
                        help="Use dummy IL policy for testing")
    parser.add_argument("--training-data-num", type=int, default=100,
                        help="Training data config for SADP_G")
    parser.add_argument("--stage-1-checkpoint", type=int, default=1500,
                        help="Checkpoint for stage 1")
    parser.add_argument("--stage-2-checkpoint", type=int, default=1500,
                        help="Checkpoint for stage 2")
    parser.add_argument("--stage-3-checkpoint", type=int, default=1500,
                        help="Checkpoint for stage 3")
    
    # Residual settings (SIMPLIFIED)
    parser.add_argument("--arm-residual-scale", type=float, default=0.05,
                        help="Max residual for arm joints")
    
    # Training settings
    parser.add_argument("--total-timesteps", type=int, default=200000,
                        help="Total training timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Steps per rollout")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="PPO epochs per update")
    
    # Output settings
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/multi_stage_ppo_simple",
                        help="Directory for checkpoints")
    parser.add_argument("--log-dir", type=str, default="./logs/multi_stage_ppo_simple",
                        help="Directory for logs")
    parser.add_argument("--tensorboard-log", type=str, default="./logs/multi_stage_ppo_simple",
                        help="TensorBoard log directory")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device (cuda:0, cpu, etc.)")
    
    return parser.parse_args()


def make_env(args, rank=0):
    """Create environment."""
    def _init():
        env = MultiStageResidualEnvSimple(
            training_data_num=args.training_data_num,
            stage_1_checkpoint=args.stage_1_checkpoint,
            stage_2_checkpoint=args.stage_2_checkpoint,
            stage_3_checkpoint=args.stage_3_checkpoint,
            use_dummy_il=args.use_dummy_il,
            max_episode_steps=args.max_episode_steps,
            arm_residual_scale=args.arm_residual_scale,
            verbose=True,
            print_every_n_steps=50,
            device=args.device,
        )
        env = Monitor(env, filename=None, allow_early_resets=True)
        return env
    return _init


def main():
    args = parse_args()
    
    print("="*60)
    print("SIMPLIFIED Multi-Stage Residual PPO Training")
    print("="*60)
    print(f"  Device: {args.device}")
    print(f"  Total Timesteps: {args.total_timesteps}")
    print(f"  IL Policy: {'Dummy' if args.use_dummy_il else 'SADP_G'}")
    print(f"  Arm Residual Scale: {args.arm_residual_scale}")
    print("="*60)
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.tensorboard_log, exist_ok=True)
    
    # Create environment
    env = DummyVecEnv([make_env(args)])
    
    # Create model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"multi_stage_simple_{timestamp}"
    
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=args.tensorboard_log,
        verbose=1,
        device=args.device,
        policy_kwargs={
            "net_arch": [256, 256],
            "activation_fn": torch.nn.ReLU,
        },
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(args.checkpoint_dir, model_name),
        name_prefix="checkpoint",
    )
    
    # Train
    try:
        print(f"\n🚀 Starting training...")
        print(f"   Model: {model_name}")
        print(f"   Checkpoints: {args.checkpoint_dir}/{model_name}/")
        print(f"   TensorBoard: tensorboard --logdir {args.tensorboard_log}\n")
        
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
        )
        
        # Save final model
        final_path = os.path.join(args.checkpoint_dir, f"{model_name}_final")
        model.save(final_path)
        print(f"\n✅ Training completed!")
        print(f"   Final model saved to {final_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        final_path = os.path.join(args.checkpoint_dir, f"{model_name}_interrupted")
        model.save(final_path)
        print(f"   Model saved to {final_path}")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        final_path = os.path.join(args.checkpoint_dir, f"{model_name}_error")
        model.save(final_path)
        print(f"[INFO] Model saved to {final_path}")
    finally:
        env.close()
        print("[MultiStageResidualEnvSimple] Environment closed.")


if __name__ == "__main__":
    main()
    simulation_app.close()

