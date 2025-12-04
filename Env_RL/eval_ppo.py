#!/usr/bin/env python3
"""
Evaluation Script for trained PPO agent on DexGarmentLab Fold Tops Task.

Usage:
    python Env_RL/eval_ppo.py --model-path checkpoints/ppo_fold_tops/best/best_model.zip --n-episodes 10
"""

# IMPORTANT: SimulationApp must be created BEFORE any other Isaac Sim imports
from isaacsim import SimulationApp

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO on FoldTops task")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model (.zip file)"
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
        help="Run in headless mode"
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record evaluation videos"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="./eval_videos",
        help="Directory to save videos"
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
        "--garment-usd",
        type=str,
        default=None,
        help="Path to garment USD file"
    )
    parser.add_argument(
        "--use-gam",
        action="store_true",
        default=True,
        help="Use GAM features"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed episode information"
    )
    
    return parser.parse_args()


# Parse args and create SimulationApp
args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

import os
import sys
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from stable_baselines3 import PPO
except ImportError:
    print("ERROR: stable-baselines3 not installed!")
    print("Please install with: pip install stable-baselines3[extra]")
    simulation_app.close()
    sys.exit(1)

from Env_RL.fold_tops_gym_env import FoldTopsGymEnv


def evaluate_agent(
    model_path: str,
    env: FoldTopsGymEnv,
    n_episodes: int = 10,
    deterministic: bool = True,
    verbose: bool = False,
    record_video: bool = False,
    video_dir: str = "./eval_videos",
):
    """
    Evaluate a trained PPO agent.
    
    Args:
        model_path: Path to the trained model
        env: The evaluation environment
        n_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
        verbose: Print detailed info
        record_video: Whether to record videos
        video_dir: Directory to save videos
        
    Returns:
        Dictionary with evaluation statistics
    """
    
    # Load the trained model
    print(f"[INFO] Loading model from {model_path}")
    model = PPO.load(model_path, env=env)
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    episode_infos = []
    
    # Video recording setup
    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        video_frames = []
    
    print(f"\n[INFO] Running {n_episodes} evaluation episodes...")
    print("-" * 50)
    
    for episode in range(n_episodes):
        obs, info = env.reset(seed=args.seed + episode)
        done = False
        episode_reward = 0.0
        episode_length = 0
        episode_info = {
            "rewards": [],
            "actions": [],
        }
        
        if record_video:
            video_frames = []
        
        while not done:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track statistics
            episode_reward += reward
            episode_length += 1
            episode_info["rewards"].append(reward)
            episode_info["actions"].append(action.copy())
            
            # Record video frame
            if record_video:
                frame = env.render()
                if frame is not None:
                    video_frames.append(frame)
            
            if verbose:
                print(f"  Step {episode_length}: reward={reward:.4f}, "
                      f"total={episode_reward:.4f}")
        
        # Episode complete
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_infos.append(episode_info)
        
        if info.get("is_success", False):
            success_count += 1
        
        print(f"Episode {episode + 1}/{n_episodes}: "
              f"reward={episode_reward:.2f}, "
              f"length={episode_length}, "
              f"success={info.get('is_success', False)}")
        
        # Save video for this episode
        if record_video and len(video_frames) > 0:
            video_path = os.path.join(video_dir, f"episode_{episode + 1}.mp4")
            save_video(video_frames, video_path)
            print(f"  Video saved to {video_path}")
    
    print("-" * 50)
    
    # Compute statistics
    stats = {
        "n_episodes": n_episodes,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "success_rate": success_count / n_episodes,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }
    
    return stats


def save_video(frames, path, fps=30):
    """Save frames as MP4 video."""
    try:
        import cv2
        
        if len(frames) == 0:
            return
            
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        out.release()
    except ImportError:
        print("[WARNING] OpenCV not installed, cannot save video")


def main():
    """Main evaluation function."""
    
    print("=" * 60)
    print("PPO Evaluation for DexGarmentLab Fold Tops Task")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Number of episodes: {args.n_episodes}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Headless: {args.headless}")
    print("=" * 60)
    
    # Check model path exists
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model not found at {args.model_path}")
        simulation_app.close()
        return
    
    # Environment configuration
    env_config = {
        "usd_path": args.garment_usd,
        "ground_material_usd": None,
    }
    
    # Create environment
    print("\n[INFO] Creating evaluation environment...")
    env = FoldTopsGymEnv(
        config=env_config,
        render_mode="human" if not args.headless else "rgb_array",
        max_episode_steps=300,
        action_scale=0.05,
        use_gam_features=args.use_gam,
        point_cloud_size=2048,
    )
    
    # Run evaluation
    try:
        stats = evaluate_agent(
            model_path=args.model_path,
            env=env,
            n_episodes=args.n_episodes,
            deterministic=args.deterministic,
            verbose=args.verbose,
            record_video=args.record_video,
            video_dir=args.video_dir,
        )
        
        # Print results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Episodes:      {stats['n_episodes']}")
        print(f"Mean Reward:   {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Min Reward:    {stats['min_reward']:.2f}")
        print(f"Max Reward:    {stats['max_reward']:.2f}")
        print(f"Mean Length:   {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
        print(f"Success Rate:  {stats['success_rate'] * 100:.1f}%")
        print("=" * 60)
        
        # Save results
        results_path = os.path.join(
            os.path.dirname(args.model_path),
            f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        )
        np.savez(results_path, **stats)
        print(f"\n[INFO] Results saved to {results_path}")
        
    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted by user.")
    finally:
        env.close()
        simulation_app.close()
    
    print("\n[INFO] Evaluation complete!")


if __name__ == "__main__":
    main()

