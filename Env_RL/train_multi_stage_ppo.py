#!/usr/bin/env python3
"""
Multi-Stage Residual PPO Training Script with SADP_G Guidance.

This script trains a single RL policy that:
1. Learns residual corrections to 3 SADP_G stage models
2. Learns WHEN to transition between stages
3. Handles all 3 folding stages with one policy

Architecture:
    action = [joint_residuals(60D), stage_advance(1D)]
    final_joint_action = SADP_G_stage[current](obs) + joint_residuals
    
    NOTE: Actions are in JOINT SPACE (60D) to match SADP_G outputs exactly!

Key Features:
- Single policy handles all stages (stage-conditioned via observation)
- Stage transitions are learned, not hard-coded
- IL provides strong baseline, RL learns corrections
- Bounded residuals ensure stability

Usage:
    # Test with dummy IL policy (no checkpoints needed)
    python Env_RL/train_multi_stage_ppo.py --use-dummy-il
    
    # With actual SADP_G checkpoints
    python Env_RL/train_multi_stage_ppo.py \\
        --training-data-num 100 \\
        --stage-1-checkpoint 1500 \\
        --stage-2-checkpoint 1500 \\
        --stage-3-checkpoint 1500

Requirements:
    - stable-baselines3
    - tensorboard
    - gymnasium
"""

from isaacsim import SimulationApp
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Stage Residual PPO")
    
    # Environment settings
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--max-episode-steps", type=int, default=400,
                        help="Maximum steps per episode (across all stages)")
    
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
    
    # Residual settings (separate for arm vs hand - hand is more sensitive!)
    parser.add_argument("--arm-residual-scale", type=float, default=0.005,
                        help="Max residual for arm joints (6 DOF per arm)")
    parser.add_argument("--hand-residual-scale", type=float, default=0.001,
                        help="Max residual for hand joints (24 DOF per hand, more sensitive!)")
    parser.add_argument("--action-scale", type=float, default=0.05,
                        help="Scale for robot commands")
    
    # Stage transition settings
    parser.add_argument("--stage-advance-threshold", type=float, default=0.5,
                        help="Threshold for stage advance signal")
    parser.add_argument("--min-steps-before-advance", type=int, default=10,
                        help="Minimum steps in stage before can advance")
    
    # ========== Quick Wins: Surgical Residual Application ==========
    parser.add_argument("--arm-only-residuals", action="store_true", default=True,
                        help="Only apply residuals to arm joints (hand = pure IL)")
    parser.add_argument("--no-arm-only-residuals", action="store_false", dest="arm_only_residuals",
                        help="Apply residuals to all joints including hand")
    parser.add_argument("--residual-apply-interval", type=int, default=5,
                        help="Apply residuals every N steps (sparse intervention)")
    parser.add_argument("--disable-manipulation-residuals", action="store_true", default=True,
                        help="Skip residuals when gripper is closed/grasping")
    parser.add_argument("--no-disable-manipulation-residuals", action="store_false", 
                        dest="disable_manipulation_residuals",
                        help="Apply residuals even during manipulation")
    parser.add_argument("--gripper-threshold", type=float, default=0.5,
                        help="Threshold for detecting gripper closed state")
    
    # ========== Initial State Configuration ==========
    # Fix distribution shift: match SADP_G training distribution!
    parser.add_argument("--use-fixed-initial-state", action="store_true", default=True,
                        help="Use fixed garment position (matches SADP_G validation)")
    parser.add_argument("--use-random-initial-state", action="store_false", dest="use_fixed_initial_state",
                        help="Use random garment position (matches SADP_G training with randomization)")
    parser.add_argument("--fixed-pos-x", type=float, default=0.0,
                        help="Fixed X position for garment (default 0.0 = center)")
    parser.add_argument("--fixed-pos-y", type=float, default=0.8,
                        help="Fixed Y position for garment (default 0.8)")
    
    # Training settings
    parser.add_argument("--total-timesteps", type=int, default=500000,
                        help="Total training timesteps")
    parser.add_argument("--save-freq", type=int, default=20000,
                        help="Save checkpoint every N timesteps")
    parser.add_argument("--eval-freq", type=int, default=40000,
                        help="Evaluate every N timesteps")
    parser.add_argument("--n-eval-episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    
    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512,
                        help="Steps per update (larger for multi-stage)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.005,
                        help="Entropy coefficient (higher for more exploration)")
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    
    # Logging
    parser.add_argument("--log-dir", type=str, default="./logs/multi_stage_ppo")
    parser.add_argument("--save-dir", type=str, default="./checkpoints/multi_stage_ppo")
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
        BaseCallback,
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

from Env_RL.multi_stage_residual_env import MultiStageResidualEnv


class MultiStagePolicyNetwork:
    """
    Custom policy network for multi-stage residual RL.
    
    Key design choices:
    - Stage-aware: processes stage indicators
    - IL-aware: processes proposed IL action (60D joint space)
    - Outputs residuals in same joint space as SADP_G
    - Separate output for stage advance decision
    """
    
    @staticmethod
    def get_policy_kwargs():
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        import torch
        import torch.nn as nn
        
        class MultiStageFeatureExtractor(BaseFeaturesExtractor):
            """Feature extractor for multi-stage residual policy (joint-space)."""
            
            def __init__(self, observation_space, features_dim: int = 256):
                super().__init__(observation_space, features_dim)
                
                # Point cloud encoder
                self.pcd_encoder = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(2048 * 3, 512),
                    nn.ReLU(),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                )
                
                # Joint position encoder (current state)
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
                
                # IL action encoder (NOW 60D joint space - same as SADP_G output!)
                # This is critical: policy sees what IL wants to do
                self.il_action_encoder = nn.Sequential(
                    nn.Linear(60, 64),  # 60D joint space
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                )
                
                # GAM keypoints encoder
                self.gam_encoder = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(6 * 3, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                )
                
                # Stage indicator encoder (critical for stage-aware behavior)
                self.stage_encoder = nn.Sequential(
                    nn.Linear(4 + 1 + 3, 32),  # one_hot(4) + progress(1) + completed(3)
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                )
                
                # Total: 128 + 32 + 16 + 32 + 16 + 16 = 240
                combined_dim = 128 + 32 + 16 + 32 + 16 + 16
                
                # Final combiner
                self.combiner = nn.Sequential(
                    nn.Linear(combined_dim, features_dim),
                    nn.ReLU(),
                )
                
            def forward(self, observations):
                # Encode each observation component
                pcd_features = self.pcd_encoder(observations["garment_pcd"])
                joint_features = self.joint_encoder(observations["joint_positions"])
                ee_features = self.ee_encoder(observations["ee_poses"])
                il_action_features = self.il_action_encoder(observations["il_action"])
                gam_features = self.gam_encoder(observations["gam_keypoints"])
                
                # Combine stage information
                stage_info = torch.cat([
                    observations["current_stage"],
                    observations["stage_progress"],
                    observations["stages_completed"],
                ], dim=-1)
                stage_features = self.stage_encoder(stage_info)
                
                # Combine all features
                combined = torch.cat([
                    pcd_features,
                    joint_features,
                    ee_features,
                    il_action_features,
                    gam_features,
                    stage_features,
                ], dim=1)
                
                return self.combiner(combined)
        
        return {
            "features_extractor_class": MultiStageFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": dict(pi=[128, 64], vf=[128, 64]),
        }


class StageTransitionCallback(BaseCallback):
    """
    Callback to log stage transition statistics.
    
    Tracks:
    - Stage advance frequency
    - Success rate per stage
    - Transition timing
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.stage_advances = {1: 0, 2: 0, 3: 0}
        self.premature_advances = 0
        self.successful_completions = 0
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        # Check if any environment finished an episode
        for info in self.locals.get("infos", []):
            if "stage_advanced" in info and info["stage_advanced"]:
                stage_info = info.get("stage_info", {})
                prev_stage = stage_info.get("stage", 1) - 1
                if prev_stage > 0:
                    self.stage_advances[prev_stage] = self.stage_advances.get(prev_stage, 0) + 1
            
            if "episode" in info:
                self.episode_count += 1
                if info.get("is_success", False):
                    self.successful_completions += 1
        
        # Log periodically
        if self.n_calls % 1000 == 0 and self.episode_count > 0:
            self.logger.record("stage/advances_stage1", self.stage_advances.get(1, 0))
            self.logger.record("stage/advances_stage2", self.stage_advances.get(2, 0))
            self.logger.record("stage/advances_stage3", self.stage_advances.get(3, 0))
            self.logger.record("stage/success_rate", 
                             self.successful_completions / max(1, self.episode_count))
        
        return True


def main():
    print("=" * 70)
    print("Multi-Stage Residual PPO Training with SADP_G Guidance")
    print("=" * 70)
    print(f"IL Policy: {'Dummy (testing)' if args.use_dummy_il else 'SADP_G'}")
    if not args.use_dummy_il:
        print(f"  Stage 1 Checkpoint: {args.stage_1_checkpoint}")
        print(f"  Stage 2 Checkpoint: {args.stage_2_checkpoint}")
        print(f"  Stage 3 Checkpoint: {args.stage_3_checkpoint}")
        print(f"  Training Data: {args.training_data_num}")
    print("-" * 70)
    print("Quick Wins: Surgical Residual Application")
    print(f"  Arm-Only Residuals: {args.arm_only_residuals} (hand = pure IL)")
    print(f"  Residual Apply Interval: every {args.residual_apply_interval} steps")
    print(f"  Skip During Manipulation: {args.disable_manipulation_residuals}")
    print(f"  Gripper Threshold: {args.gripper_threshold}")
    print("-" * 70)
    print("Initial State Configuration (Fix Distribution Shift!)")
    if args.use_fixed_initial_state:
        print(f"  Mode: FIXED (matches SADP_G validation)")
        print(f"  Position: X={args.fixed_pos_x}, Y={args.fixed_pos_y}")
    else:
        print(f"  Mode: RANDOM (matches SADP_G training with randomization)")
        print(f"  X range: [-0.1, 0.1], Y range: [0.7, 0.9]")
    print("-" * 70)
    print(f"Arm Residual Scale: ±{args.arm_residual_scale} (6 DOF per arm)")
    print(f"Hand Residual Scale: ±{args.hand_residual_scale} (24 DOF per hand, smaller!)")
    print(f"Stage Advance Threshold: {args.stage_advance_threshold}")
    print(f"Headless: {args.headless}")
    print(f"Total Timesteps: {args.total_timesteps}")
    print("=" * 70)
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.exp_name:
        run_name = f"multi_stage_{args.exp_name}_{timestamp}"
    else:
        il_type = "dummy" if args.use_dummy_il else "sadpg"
        run_name = f"multi_stage_{il_type}_{timestamp}"
    
    # Create environment
    print("\n[INFO] Creating Multi-Stage Residual RL environment...")
    env = MultiStageResidualEnv(
        training_data_num=args.training_data_num,
        stage_1_checkpoint=args.stage_1_checkpoint,
        stage_2_checkpoint=args.stage_2_checkpoint,
        stage_3_checkpoint=args.stage_3_checkpoint,
        use_dummy_il=args.use_dummy_il,
        render_mode="human" if not args.headless else "rgb_array",
        max_episode_steps=args.max_episode_steps,
        arm_residual_scale=args.arm_residual_scale,
        hand_residual_scale=args.hand_residual_scale,
        action_scale=args.action_scale,
        # Quick Wins: Surgical Residual Application
        arm_only_residuals=args.arm_only_residuals,
        residual_apply_interval=args.residual_apply_interval,
        disable_residual_during_manipulation=args.disable_manipulation_residuals,
        gripper_threshold=args.gripper_threshold,
        # Initial State: Fix distribution shift!
        use_fixed_initial_state=args.use_fixed_initial_state,
        fixed_pos_x=args.fixed_pos_x,
        fixed_pos_y=args.fixed_pos_y,
        # Stage transition
        stage_advance_threshold=args.stage_advance_threshold,
        min_steps_before_advance=args.min_steps_before_advance,
    )
    env = Monitor(env, filename=os.path.join(args.log_dir, run_name))
    
    # Get policy kwargs
    policy_kwargs = MultiStagePolicyNetwork.get_policy_kwargs()
    
    # Create model
    if args.load_path and os.path.exists(args.load_path):
        print(f"\n[INFO] Loading model from {args.load_path}")
        model = PPO.load(args.load_path, env=env)
    else:
        print("\n[INFO] Creating new Multi-Stage Residual PPO model...")
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
        
        # Initialize policy output layer with small weights
        # This ensures initial residuals are near-zero (IL dominates early)
        print("\n[INFO] Initializing policy output layer with small weights...")
        import torch.nn as nn
        def init_small_output_weights(module):
            """Initialize output layer with small weights for near-zero initial residuals."""
            if isinstance(module, nn.Linear):
                # Check if this is likely an output layer (61 outputs = 60 joint residuals + 1 stage advance)
                if module.out_features == 61:
                    nn.init.uniform_(module.weight, -0.001, 0.001)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                    print(f"    Initialized output layer: {module.in_features} -> {module.out_features}")
        
        # Apply to action network
        model.policy.action_net.apply(init_small_output_weights)
        model.policy.mlp_extractor.policy_net.apply(init_small_output_weights)
        print("[INFO] Policy output initialization complete!")
    
    # Setup callbacks
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
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )
    
    stage_cb = StageTransitionCallback(verbose=1)
    
    callbacks = CallbackList([checkpoint_cb, eval_cb, stage_cb])
    
    # Configure logger
    logger = configure(args.log_dir, ["stdout", "tensorboard"])
    model.set_logger(logger)
    
    # Print model summary
    print("\n[INFO] Model Summary:")
    print(f"  Policy: MultiInputPolicy with MultiStageFeatureExtractor")
    print(f"  Action Space: Box(61,) - [joint_residuals(60D), stage_advance(1D)]")
    print(f"  Observation includes: garment_pcd, joints, ee_poses, il_action(60D),")
    print(f"                        gam_keypoints, current_stage, stage_progress")
    print(f"  NOTE: Actions are in JOINT SPACE to match SADP_G outputs!")
    print(f"  Device: {model.device}")
    
    # Start training
    print("\n" + "=" * 70)
    print("Starting Multi-Stage Residual RL Training")
    print("-" * 70)
    print("Architecture (SURGICAL INTERVENTION):")
    print("  final_action = IL_action + surgical_residual")
    print("  ")
    if args.arm_only_residuals:
        print("  ARM-ONLY: Residuals on 12 arm joints, hand (48 joints) = pure IL")
    else:
        print("  FULL: Residuals on all 60 joints")
    print(f"  SPARSE: Residuals applied every {args.residual_apply_interval} steps")
    if args.disable_manipulation_residuals:
        print("  PROTECTED: No residuals during manipulation (gripper closed)")
    print("  ")
    print("  RL makes tiny, surgical improvements to working IL policy!")
    print("-" * 70)
    print(f"Monitor with: tensorboard --logdir={args.log_dir}")
    print("=" * 70 + "\n")
    
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
        print("\n[INFO] Training interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save final model
        final_path = os.path.join(args.save_dir, f"{run_name}_final")
        model.save(final_path)
        print(f"\n[INFO] Final model saved to {final_path}")
    
    # Cleanup
    env.close()
    simulation_app.close()
    
    print("\n[INFO] Training complete!")
    print(f"  Logs: {args.log_dir}")
    print(f"  Checkpoints: {args.save_dir}")


if __name__ == "__main__":
    main()
