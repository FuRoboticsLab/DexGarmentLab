#!/usr/bin/env python3
"""
Evaluation Script for Hierarchical PPO on Garment Folding.

This script evaluates a trained hierarchical policy and visualizes
the primitive sequence it learns to execute.

Usage:
    python Env_RL/eval_hierarchical.py --model-path checkpoints/hierarchical_ppo/best/best_model.zip
"""

from isaacsim import SimulationApp
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Hierarchical PPO")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headless"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed info"
    )
    
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

import os
import sys
import numpy as np
from datetime import datetime
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from stable_baselines3 import PPO
except ImportError:
    print("ERROR: stable-baselines3 not installed!")
    simulation_app.close()
    sys.exit(1)

from Env_RL.hierarchical_fold_env import HierarchicalFoldEnv
from Env_RL.primitives import PrimitiveID


def evaluate(model_path, env, n_episodes, deterministic=True, verbose=False):
    """
    Evaluate the trained hierarchical policy.
    
    Returns statistics about:
    - Success rate
    - Average episode length (primitives)
    - Most common primitive sequences
    """
    
    print(f"[INFO] Loading model from {model_path}")
    model = PPO.load(model_path, env=env)
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    successes = []
    sequences = []  # List of primitive sequences
    
    print(f"\n[INFO] Running {n_episodes} evaluation episodes...")
    print("-" * 60)
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        episode_reward = 0
        episode_length = 0
        sequence = []
        
        print(f"\nEpisode {ep + 1}/{n_episodes}")
        print("-" * 40)
        
        while not done:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=deterministic)
            primitive_name = PrimitiveID(action).name
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            sequence.append(primitive_name)
            
            # Print step info
            status = "✓" if reward > 0 else "✗" if reward < 0 else "○"
            print(f"  Step {episode_length}: {primitive_name:20s} → reward: {reward:+.2f} {status}")
            
            if verbose and "message" in info:
                print(f"           Message: {info['message']}")
        
        # Episode summary
        success = info.get("success", False)
        successes.append(success)
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        sequences.append(tuple(sequence))
        
        result = "SUCCESS ✓" if success else "FAILED ✗"
        print(f"  Result: {result}")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Sequence: {' → '.join(sequence)}")
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Compute statistics
    success_rate = sum(successes) / len(successes)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"Success Rate:     {success_rate * 100:.1f}%")
    print(f"Mean Reward:      {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Length:      {mean_length:.1f} primitives")
    
    # Most common sequences
    print("\nMost Common Sequences:")
    sequence_counts = Counter(sequences)
    for seq, count in sequence_counts.most_common(5):
        pct = count / len(sequences) * 100
        seq_str = " → ".join(seq)
        print(f"  ({count}/{len(sequences)}, {pct:.0f}%) {seq_str}")
    
    # Primitive usage statistics
    print("\nPrimitive Usage:")
    all_primitives = [p for seq in sequences for p in seq]
    primitive_counts = Counter(all_primitives)
    for primitive, count in primitive_counts.most_common():
        pct = count / len(all_primitives) * 100
        print(f"  {primitive:20s}: {count:3d} ({pct:.1f}%)")
    
    print("=" * 60)
    
    return {
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "sequences": sequences,
    }


def main():
    print("=" * 60)
    print("Hierarchical PPO Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.n_episodes}")
    print("=" * 60)
    
    # Check model exists
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model not found: {args.model_path}")
        simulation_app.close()
        return
    
    # Create environment
    print("\n[INFO] Creating environment...")
    env = HierarchicalFoldEnv(
        config={},
        render_mode="human" if not args.headless else "rgb_array",
        max_primitives=10,
    )
    
    try:
        results = evaluate(
            model_path=args.model_path,
            env=env,
            n_episodes=args.n_episodes,
            deterministic=args.deterministic,
            verbose=args.verbose,
        )
        
        # Save results
        results_path = os.path.join(
            os.path.dirname(args.model_path),
            f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        )
        np.savez(results_path, **{k: v for k, v in results.items() if k != "sequences"})
        print(f"\n[INFO] Results saved to {results_path}")
        
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        simulation_app.close()
    
    print("\n[INFO] Evaluation complete!")


if __name__ == "__main__":
    main()



