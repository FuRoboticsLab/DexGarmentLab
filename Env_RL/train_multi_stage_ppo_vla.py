#!/usr/bin/env python3
"""
VLA-Enhanced Multi-Stage Residual PPO Training Script.

This script trains a PPO policy with VLA feature extraction.
The policy learns to combine point cloud, GAM keypoints, and VLA visual features.

Usage:
    # With CLIP (lightweight, recommended)
    python Env_RL/train_multi_stage_ppo_vla.py --vla-model clip
    
    # With dummy VLA (for testing)
    python Env_RL/train_multi_stage_ppo_vla.py --vla-model dummy
    
    # Disable VLA (baseline comparison)
    python Env_RL/train_multi_stage_ppo_vla.py --disable-vla
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
import torch.nn as nn
from typing import Dict
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import configure
import gymnasium as gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Env_RL.multi_stage_residual_env_vla import MultiStageResidualEnvVLA


class VLAEnhancedFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that processes:
    - Point cloud (geometry)
    - GAM keypoints (affordances)
    - VLA visual features (semantic understanding)
    - Other state information
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        use_vla: bool = True,
    ):
        super().__init__(observation_space, features_dim)
        self.use_vla = use_vla
        
        # Point cloud encoder (2048 points × 3 → 64D)
        self.pcd_encoder = nn.Sequential(
            nn.Linear(2048 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        # GAM keypoints encoder (6 keypoints × 3 → 32D)
        self.gam_encoder = nn.Sequential(
            nn.Linear(6 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        
        # State encoder (joints + EE poses + IL action + stage info)
        state_dim = (
            60 +  # joint_positions
            14 +  # ee_poses
            60 +  # il_action
            4 +   # current_stage
            1 +   # stage_progress
            3 +   # stages_completed
            1     # should_advance_hint
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        
        # VLA encoders (if using VLA)
        if self.use_vla:
            # VLA visual features encoder (256D → 64D)
            self.vla_visual_encoder = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )
            
            # VLA stage guidance encoder (128D → 32D)
            self.vla_guidance_encoder = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
            )
            
            # Combined feature dimension
            combined_dim = 64 + 32 + 32 + 64 + 32  # pcd + gam + state + vla_visual + vla_guidance
        else:
            # Without VLA
            combined_dim = 64 + 32 + 32  # pcd + gam + state
        
        # Final combiner
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from observations."""
        batch_size = observations["garment_pcd"].shape[0]
        
        # Encode point cloud
        pcd = observations["garment_pcd"].view(batch_size, -1)  # (B, 2048*3)
        pcd_features = self.pcd_encoder(pcd)  # (B, 64)
        
        # Encode GAM keypoints
        gam = observations["gam_keypoints"].view(batch_size, -1)  # (B, 6*3)
        gam_features = self.gam_encoder(gam)  # (B, 32)
        
        # Encode state
        state = torch.cat([
            observations["joint_positions"],
            observations["ee_poses"],
            observations["il_action"],
            observations["current_stage"],
            observations["stage_progress"],
            observations["stages_completed"],
            observations["should_advance_hint"],
        ], dim=-1)  # (B, state_dim)
        state_features = self.state_encoder(state)  # (B, 32)
        
        # Combine base features
        if self.use_vla and "vla_visual_features" in observations:
            # Encode VLA visual features
            vla_visual = observations["vla_visual_features"]  # (B, 256)
            vla_visual_features = self.vla_visual_encoder(vla_visual)  # (B, 64)
            
            # Encode VLA stage guidance
            vla_guidance = observations["vla_stage_guidance"]  # (B, 128)
            vla_guidance_features = self.vla_guidance_encoder(vla_guidance)  # (B, 32)
            
            # Combine all features
            combined = torch.cat([
                pcd_features,
                gam_features,
                state_features,
                vla_visual_features,
                vla_guidance_features,
            ], dim=-1)  # (B, combined_dim)
        else:
            # Without VLA
            combined = torch.cat([
                pcd_features,
                gam_features,
                state_features,
            ], dim=-1)  # (B, combined_dim)
        
        # Final combination
        features = self.combiner(combined)  # (B, features_dim)
        
        return features


def parse_args():
    parser = argparse.ArgumentParser(description="Train VLA-Enhanced Multi-Stage Residual PPO")
    
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
    
    # Residual settings - SIMPLIFIED: Let RL learn from IL baseline
    parser.add_argument("--arm-residual-scale", type=float, default=0.2,
                        help="Max residual for arm joints (larger = more RL freedom)")
    parser.add_argument("--residual-apply-interval", type=int, default=1,
                        help="Apply residuals every N steps (1=every step)")
    parser.add_argument("--disable-manipulation-residuals", action="store_true", default=False,
                        help="Block residuals during manipulation (default: False - allow RL always)")
    
    # VLA settings [NEW]
    parser.add_argument("--vla-model", type=str, default="clip",
                        choices=["clip", "openvla", "dummy"],
                        help="VLA model to use")
    parser.add_argument("--disable-vla", action="store_true",
                        help="Disable VLA features (baseline)")
    parser.add_argument("--vla-feature-dim", type=int, default=256,
                        help="VLA visual feature dimension")
    parser.add_argument("--vla-guidance-dim", type=int, default=128,
                        help="VLA guidance feature dimension")
    
    # Stage transition control (IMPROVED DEFAULTS)
    parser.add_argument("--proactive-advance-after-ratio", type=float, default=0.85,
                        help="Auto-advance after this ratio of IL steps")
    parser.add_argument("--min-quality-for-proactive", type=float, default=0.3,
                        help="Minimum quality for proactive advance (increased from 0.1)")
    parser.add_argument("--stage-advance-threshold", type=float, default=0.5,
                        help="RL signal threshold for stage advance")
    parser.add_argument("--min-steps-before-advance", type=int, default=10,
                        help="Min steps before RL can advance")
    parser.add_argument("--quality-plateau-window", type=int, default=50,
                        help="Steps to detect quality plateau (increased from 30)")
    parser.add_argument("--quality-plateau-threshold", type=float, default=0.05,
                        help="Max improvement to be 'plateau' (increased from 0.02)")
    
    # Early termination control
    parser.add_argument("--enable-early-termination", action="store_true",
                        help="Enable early termination for bad episodes")
    parser.add_argument("--early-termination-penalty", type=float, default=5.0,
                        help="Penalty for early termination")
    
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
    parser.add_argument("--eval-freq", type=int, default=20000,
                        help="Evaluate every N timesteps")
    parser.add_argument("--n-eval-episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    
    # Output settings
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/multi_stage_ppo_vla",
                        help="Directory for checkpoints")
    parser.add_argument("--log-dir", type=str, default="./logs/multi_stage_ppo_vla",
                        help="Directory for logs")
    parser.add_argument("--tensorboard-log", type=str, default="./logs/multi_stage_ppo_vla",
                        help="TensorBoard log directory")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device (cuda:0, cpu, etc.)")
    
    return parser.parse_args()


def make_env(args, rank=0, eval=False):
    """Create environment."""
    def _init():
        env = MultiStageResidualEnvVLA(
            training_data_num=args.training_data_num,
            stage_1_checkpoint=args.stage_1_checkpoint,
            stage_2_checkpoint=args.stage_2_checkpoint,
            stage_3_checkpoint=args.stage_3_checkpoint,
            use_dummy_il=args.use_dummy_il,
            max_episode_steps=args.max_episode_steps,
            arm_residual_scale=args.arm_residual_scale,
            # RL intervention control - SIMPLIFIED
            residual_apply_interval=args.residual_apply_interval,
            enable_phase_aware_control=False,  # Disabled - let RL learn
            disable_residual_during_manipulation=args.disable_manipulation_residuals,
            # Stage transition control
            proactive_advance_after_ratio=args.proactive_advance_after_ratio,
            min_quality_for_proactive=args.min_quality_for_proactive,
            stage_advance_threshold=args.stage_advance_threshold,
            min_steps_before_advance=args.min_steps_before_advance,
            quality_plateau_window=args.quality_plateau_window,
            quality_plateau_threshold=args.quality_plateau_threshold,
            # Early termination
            enable_early_termination=args.enable_early_termination,
            early_termination_penalty=args.early_termination_penalty,
            # VLA settings
            vla_model_name=args.vla_model if not args.disable_vla else None,
            vla_feature_dim=args.vla_feature_dim,
            vla_guidance_dim=args.vla_guidance_dim,
            use_vla_features=not args.disable_vla,
            verbose=not eval,  # Less verbose during evaluation
            print_every_n_steps=50,
            device=args.device,
        )
        env = Monitor(env, filename=None, allow_early_resets=True)
        return env
    return _init


def main():
    args = parse_args()
    
    print("="*60)
    print("VLA-Enhanced Multi-Stage Residual PPO Training")
    print("="*60)
    print(f"  Device: {args.device}")
    print(f"  Total Timesteps: {args.total_timesteps}")
    print(f"  IL Policy: {'Dummy' if args.use_dummy_il else 'SADP_G'}")
    print(f"  VLA Model: {args.vla_model if not args.disable_vla else 'Disabled'}")
    print(f"  Arm Residual Scale: {args.arm_residual_scale}")
    print(f"  Residual Apply Interval: {getattr(args, 'residual_apply_interval', 1)}")
    print(f"  Min Quality for Proactive: {getattr(args, 'min_quality_for_proactive', 0.3)}")
    print(f"  Early Termination: {'Enabled' if getattr(args, 'enable_early_termination', False) else 'Disabled'}")
    print("="*60)
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.tensorboard_log, exist_ok=True)
    
    # Create environment
    env = DummyVecEnv([make_env(args)])
    
    # Determine if VLA is used (from args, more reliable)
    use_vla = not args.disable_vla and args.vla_model is not None
    
    # Create model with VLA-enhanced feature extractor
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vla_suffix = f"_{args.vla_model}" if use_vla else "_novla"
    model_name = f"multi_stage_vla{vla_suffix}_{timestamp}"
    
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
            "features_extractor_class": VLAEnhancedFeaturesExtractor,
            "features_extractor_kwargs": {
                "features_dim": 256,
                "use_vla": use_vla,
            },
            "net_arch": [256, 256],
            "activation_fn": torch.nn.ReLU,
        },
    )
    
    # Initialize policy to start near IL baseline (small residuals)
    # Set action net output to near-zero so RL starts close to IL
    with torch.no_grad():
        if hasattr(model.policy.action_net, 'weight'):
            # Initialize action output to produce small residuals
            nn.init.normal_(model.policy.action_net.weight, mean=0.0, std=0.01)
            if model.policy.action_net.bias is not None:
                nn.init.constant_(model.policy.action_net.bias, 0.0)
    print("[INFO] Policy initialized to output near-zero residuals (starts close to IL baseline)")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(args.checkpoint_dir, model_name),
        name_prefix="checkpoint",
    )
    
    # Evaluation callback (IMPROVED: Track progress and save best model)
    eval_env = DummyVecEnv([make_env(args, eval=True)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.checkpoint_dir, model_name, "best"),
        log_path=args.log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    
    # Stage transition callback (IMPROVED: Monitor stage transitions)
    class StageTransitionCallback(BaseCallback):
        """Track stage transitions and quality metrics."""
        
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.stage_transitions = []
            self.episode_stats = []
            self.current_episode_transitions = []
            self.current_episode_qualities = []
        
        def _on_step(self) -> bool:
            # Track stage transitions
            if len(self.locals.get("infos", [])) > 0:
                info = self.locals["infos"][0]
                if info.get("stage_advanced", False):
                    self.current_episode_transitions.append({
                        "step": self.num_timesteps,
                        "quality": info.get("stage_quality", 0),
                        "source": info.get("advance_source", "unknown"),
                        "stages_completed": sum(info.get("stages_completed", [])),
                    })
                
                # Track quality
                if "stage_quality" in info:
                    self.current_episode_qualities.append(info["stage_quality"])
            
            return True
        
        def _on_episode_end(self) -> bool:
            # Log episode summary
            if len(self.current_episode_transitions) > 0:
                self.stage_transitions.extend(self.current_episode_transitions)
                self.episode_stats.append({
                    "transitions": len(self.current_episode_transitions),
                    "avg_quality": np.mean(self.current_episode_qualities) if self.current_episode_qualities else 0,
                    "max_quality": np.max(self.current_episode_qualities) if self.current_episode_qualities else 0,
                })
            
            self.current_episode_transitions = []
            self.current_episode_qualities = []
            return True
    
    stage_callback = StageTransitionCallback(verbose=1)
    
    # Combine all callbacks
    callbacks = CallbackList([checkpoint_callback, eval_callback, stage_callback])
    
    # Configure logger (IMPROVED: Proper logging setup)
    logger = configure(args.tensorboard_log, ["stdout", "tensorboard"])
    model.set_logger(logger)
    
    # Train
    try:
        print(f"\n🚀 Starting training...")
        print(f"   Model: {model_name}")
        print(f"   Checkpoints: {args.checkpoint_dir}/{model_name}/")
        print(f"   TensorBoard: tensorboard --logdir {args.tensorboard_log}\n")
        
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
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
        eval_env.close()
        print("[MultiStageResidualEnvVLA] Environment closed.")
        
        # Print training summary
        if hasattr(stage_callback, 'episode_stats') and len(stage_callback.episode_stats) > 0:
            print("\n" + "="*60)
            print("Training Summary")
            print("="*60)
            avg_transitions = np.mean([s["transitions"] for s in stage_callback.episode_stats])
            avg_quality = np.mean([s["avg_quality"] for s in stage_callback.episode_stats])
            print(f"  Average stage transitions per episode: {avg_transitions:.2f}")
            print(f"  Average stage quality: {avg_quality:.3f}")
            print("="*60)


if __name__ == "__main__":
    main()
    simulation_app.close()

