#!/usr/bin/env python3
"""
Evaluation Script for Multi-Stage Residual PPO Policy.

This script evaluates a trained multi-stage policy on the garment folding task.
It provides detailed metrics on:
- Overall success rate
- Stage-specific performance
- Residual magnitudes
- Stage transition timing

Usage:
    # Basic evaluation
    python Env_RL/eval_multi_stage.py --model-path checkpoints/multi_stage_ppo/best/best_model.zip
    
    # Evaluation with video recording
    python Env_RL/eval_multi_stage.py --model-path PATH --record-video --n-episodes 10
    
    # Compare with IL-only baseline
    python Env_RL/eval_multi_stage.py --model-path PATH --compare-il
"""

from isaacsim import SimulationApp
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Stage Residual PPO")
    
    # Model settings
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model")
    
    # Environment settings
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=400)
    
    # SADP_G settings (should match training)
    parser.add_argument("--use-dummy-il", action="store_true")
    parser.add_argument("--training-data-num", type=int, default=100)
    parser.add_argument("--stage-1-checkpoint", type=int, default=1500)
    parser.add_argument("--stage-2-checkpoint", type=int, default=1500)
    parser.add_argument("--stage-3-checkpoint", type=int, default=1500)
    
    # Residual settings
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--stage-advance-threshold", type=float, default=0.5)
    
    # Evaluation settings
    parser.add_argument("--n-episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic policy")
    parser.add_argument("--seed", type=int, default=42)
    
    # Output settings
    parser.add_argument("--record-video", action="store_true",
                        help="Record evaluation videos")
    parser.add_argument("--output-dir", type=str, default="./eval_results/multi_stage",
                        help="Directory for evaluation outputs")
    parser.add_argument("--compare-il", action="store_true",
                        help="Also evaluate IL-only baseline for comparison")
    
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

import os
import sys
import json
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from stable_baselines3 import PPO
except ImportError:
    print("ERROR: stable-baselines3 not installed!")
    simulation_app.close()
    sys.exit(1)

from Env_RL.multi_stage_residual_env import MultiStageResidualEnv


class EvaluationMetrics:
    """Tracks evaluation metrics across episodes."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.episodes = []
        self.current_episode = {
            "steps": 0,
            "total_reward": 0.0,
            "stage_rewards": {1: 0.0, 2: 0.0, 3: 0.0},
            "stages_completed": [],
            "stage_transitions": [],  # [(step, from_stage, to_stage), ...]
            "residual_magnitudes": [],
            "success": False,
            "final_fold_quality": 0.0,
        }
    
    def step(self, info: dict, residual: np.ndarray, reward: float):
        """Record step information."""
        self.current_episode["steps"] += 1
        self.current_episode["total_reward"] += reward
        
        # Track residual magnitude
        residual_mag = np.linalg.norm(residual[:8])
        self.current_episode["residual_magnitudes"].append(residual_mag)
        
        # Track stage transitions
        if info.get("stage_advanced", False):
            stage_info = info.get("stage_info", {})
            current_stage = stage_info.get("stage", 1)
            self.current_episode["stage_transitions"].append({
                "step": self.current_episode["steps"],
                "to_stage": current_stage,
            })
        
        # Track reward per stage
        stage_info = info.get("stage_info", {})
        current_stage = stage_info.get("stage", 1)
        if current_stage <= 3:
            self.current_episode["stage_rewards"][current_stage] += reward
    
    def end_episode(self, info: dict, final_obs: dict):
        """Finalize episode and compute final metrics."""
        self.current_episode["success"] = info.get("is_success", False)
        self.current_episode["stages_completed"] = info.get("stages_completed", [False, False, False])
        self.current_episode["final_fold_quality"] = info.get("fold_progress", 0.0)
        
        # Compute average residual
        if self.current_episode["residual_magnitudes"]:
            self.current_episode["avg_residual"] = np.mean(self.current_episode["residual_magnitudes"])
            self.current_episode["max_residual"] = np.max(self.current_episode["residual_magnitudes"])
        else:
            self.current_episode["avg_residual"] = 0.0
            self.current_episode["max_residual"] = 0.0
        
        self.episodes.append(self.current_episode.copy())
        self.reset()
    
    def get_summary(self) -> dict:
        """Get summary statistics across all episodes."""
        if not self.episodes:
            return {}
        
        n_episodes = len(self.episodes)
        
        # Success metrics
        successes = sum(1 for ep in self.episodes if ep["success"])
        success_rate = successes / n_episodes
        
        # Stage completion metrics
        stage_completion_counts = [0, 0, 0]
        for ep in self.episodes:
            for i, completed in enumerate(ep["stages_completed"]):
                if completed:
                    stage_completion_counts[i] += 1
        
        # Reward metrics
        total_rewards = [ep["total_reward"] for ep in self.episodes]
        
        # Residual metrics
        avg_residuals = [ep["avg_residual"] for ep in self.episodes]
        
        # Episode length
        steps = [ep["steps"] for ep in self.episodes]
        
        # Stage transition timing
        stage_1_to_2_steps = []
        stage_2_to_3_steps = []
        for ep in self.episodes:
            for trans in ep["stage_transitions"]:
                if trans["to_stage"] == 2:
                    stage_1_to_2_steps.append(trans["step"])
                elif trans["to_stage"] == 3:
                    stage_2_to_3_steps.append(trans["step"])
        
        return {
            "n_episodes": n_episodes,
            "success_rate": success_rate,
            "stage_1_completion_rate": stage_completion_counts[0] / n_episodes,
            "stage_2_completion_rate": stage_completion_counts[1] / n_episodes,
            "stage_3_completion_rate": stage_completion_counts[2] / n_episodes,
            "mean_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "mean_episode_length": np.mean(steps),
            "std_episode_length": np.std(steps),
            "mean_residual_magnitude": np.mean(avg_residuals),
            "mean_stage_1_to_2_step": np.mean(stage_1_to_2_steps) if stage_1_to_2_steps else None,
            "mean_stage_2_to_3_step": np.mean(stage_2_to_3_steps) if stage_2_to_3_steps else None,
        }


def evaluate_rl_policy(env, model, n_episodes: int, deterministic: bool = True) -> EvaluationMetrics:
    """Evaluate the trained RL policy."""
    metrics = EvaluationMetrics()
    
    for episode in range(n_episodes):
        print(f"\n[Eval] Episode {episode + 1}/{n_episodes}")
        
        obs, info = env.reset()
        done = False
        
        while not done:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record metrics
            metrics.step(info, action, reward)
            
            # Print stage info periodically
            stage_info = info.get("stage_info", {})
            if info.get("stage_advanced", False):
                print(f"  Stage advanced to {stage_info.get('stage', '?')}")
        
        # Finalize episode
        metrics.end_episode(info, obs)
        
        # Print episode result
        ep_data = metrics.episodes[-1]
        status = "SUCCESS" if ep_data["success"] else "FAIL"
        print(f"  Result: {status}, Reward: {ep_data['total_reward']:.2f}, "
              f"Steps: {ep_data['steps']}, Stages: {ep_data['stages_completed']}")
    
    return metrics


def evaluate_il_only(env, n_episodes: int) -> EvaluationMetrics:
    """
    Evaluate IL policy only (no RL residuals).
    
    This provides a baseline to compare against.
    """
    metrics = EvaluationMetrics()
    
    for episode in range(n_episodes):
        print(f"\n[IL-Only Eval] Episode {episode + 1}/{n_episodes}")
        
        obs, info = env.reset()
        done = False
        step_count = 0
        
        while not done:
            # Use zero residuals - IL action only
            # Also periodically advance stages based on step count
            action = np.zeros(9, dtype=np.float32)
            
            # Simple stage advance heuristic (after N steps)
            stage_info = info.get("stage_info", {})
            current_stage = stage_info.get("stage", 1)
            stage_steps = stage_info.get("step_count", 0)
            max_steps = stage_info.get("max_steps", 8)
            
            if stage_steps >= max_steps:
                action[8] = 1.0  # Advance stage
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
            
            # Record metrics
            metrics.step(info, action, reward)
            
            if info.get("stage_advanced", False):
                print(f"  Stage advanced to {info.get('stage_info', {}).get('stage', '?')}")
        
        # Finalize episode
        metrics.end_episode(info, obs)
        
        ep_data = metrics.episodes[-1]
        status = "SUCCESS" if ep_data["success"] else "FAIL"
        print(f"  Result: {status}, Reward: {ep_data['total_reward']:.2f}, Steps: {ep_data['steps']}")
    
    return metrics


def main():
    print("=" * 70)
    print("Multi-Stage Residual PPO Evaluation")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Compare IL: {args.compare_il}")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create environment
    print("\n[INFO] Creating environment...")
    env = MultiStageResidualEnv(
        training_data_num=args.training_data_num,
        stage_1_checkpoint=args.stage_1_checkpoint,
        stage_2_checkpoint=args.stage_2_checkpoint,
        stage_3_checkpoint=args.stage_3_checkpoint,
        use_dummy_il=args.use_dummy_il,
        render_mode="human" if not args.headless else "rgb_array",
        max_episode_steps=args.max_episode_steps,
        residual_scale=args.residual_scale,
        stage_advance_threshold=args.stage_advance_threshold,
    )
    
    # Load model
    print(f"\n[INFO] Loading model from {args.model_path}...")
    model = PPO.load(args.model_path, env=env)
    
    # Set seed
    np.random.seed(args.seed)
    
    # Evaluate RL policy
    print("\n" + "=" * 70)
    print("Evaluating RL + IL Residual Policy")
    print("=" * 70)
    
    rl_metrics = evaluate_rl_policy(
        env=env,
        model=model,
        n_episodes=args.n_episodes,
        deterministic=args.deterministic,
    )
    
    rl_summary = rl_metrics.get_summary()
    
    # Print RL results
    print("\n" + "-" * 50)
    print("RL + IL Policy Results:")
    print("-" * 50)
    print(f"  Success Rate: {rl_summary['success_rate']*100:.1f}%")
    print(f"  Stage 1 Completion: {rl_summary['stage_1_completion_rate']*100:.1f}%")
    print(f"  Stage 2 Completion: {rl_summary['stage_2_completion_rate']*100:.1f}%")
    print(f"  Stage 3 Completion: {rl_summary['stage_3_completion_rate']*100:.1f}%")
    print(f"  Mean Reward: {rl_summary['mean_reward']:.2f} ± {rl_summary['std_reward']:.2f}")
    print(f"  Mean Episode Length: {rl_summary['mean_episode_length']:.1f} ± {rl_summary['std_episode_length']:.1f}")
    print(f"  Mean Residual Magnitude: {rl_summary['mean_residual_magnitude']:.4f}")
    if rl_summary['mean_stage_1_to_2_step']:
        print(f"  Mean Stage 1→2 Transition: Step {rl_summary['mean_stage_1_to_2_step']:.1f}")
    if rl_summary['mean_stage_2_to_3_step']:
        print(f"  Mean Stage 2→3 Transition: Step {rl_summary['mean_stage_2_to_3_step']:.1f}")
    
    results = {"rl_policy": rl_summary}
    
    # Optionally evaluate IL-only baseline
    if args.compare_il:
        print("\n" + "=" * 70)
        print("Evaluating IL-Only Baseline (Zero Residuals)")
        print("=" * 70)
        
        np.random.seed(args.seed)  # Reset seed for fair comparison
        
        il_metrics = evaluate_il_only(
            env=env,
            n_episodes=args.n_episodes,
        )
        
        il_summary = il_metrics.get_summary()
        
        # Print IL results
        print("\n" + "-" * 50)
        print("IL-Only Policy Results:")
        print("-" * 50)
        print(f"  Success Rate: {il_summary['success_rate']*100:.1f}%")
        print(f"  Stage 1 Completion: {il_summary['stage_1_completion_rate']*100:.1f}%")
        print(f"  Stage 2 Completion: {il_summary['stage_2_completion_rate']*100:.1f}%")
        print(f"  Stage 3 Completion: {il_summary['stage_3_completion_rate']*100:.1f}%")
        print(f"  Mean Reward: {il_summary['mean_reward']:.2f} ± {il_summary['std_reward']:.2f}")
        
        results["il_only"] = il_summary
        
        # Print comparison
        print("\n" + "=" * 70)
        print("Comparison: RL+IL vs IL-Only")
        print("=" * 70)
        print(f"  Success Rate Improvement: "
              f"{(rl_summary['success_rate'] - il_summary['success_rate'])*100:+.1f}%")
        print(f"  Reward Improvement: "
              f"{rl_summary['mean_reward'] - il_summary['mean_reward']:+.2f}")
    
    # Save results
    results_path = os.path.join(args.output_dir, f"eval_results_{timestamp}.json")
    with open(results_path, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump({k: {kk: convert(vv) for kk, vv in v.items()} 
                   for k, v in results.items()}, f, indent=2)
    
    print(f"\n[INFO] Results saved to {results_path}")
    
    # Cleanup
    env.close()
    simulation_app.close()
    
    print("\n[INFO] Evaluation complete!")


if __name__ == "__main__":
    main()
