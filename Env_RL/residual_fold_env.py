"""
Residual RL Environment for Garment Folding.

This environment implements Residual RL where:
- IL Policy (frozen): Provides base actions from pre-trained model
- RL Policy (trainable): Learns small corrections (residuals)
- Final Action = IL_action + RL_residual

The RL policy only needs to learn corrections to the IL policy's behavior,
which is much easier than learning from scratch.

Key Features:
- IL policy is frozen (no gradient flow)
- Residuals are bounded (small corrections only)
- Observation includes IL's proposed action
- Works with SADP, DP3, or any IL policy

Usage:
    env = ResidualFoldEnv(
        il_policy_type="SADP",
        task_name="Fold_Tops",
        checkpoint_num=1000,
    )
    obs, info = env.reset()
    action = rl_policy(obs)  # This is the RESIDUAL
    next_obs, reward, done, truncated, info = env.step(action)
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Env_RL.il_policy_wrapper import ILPolicyWrapper, DummyILPolicy, create_il_policy


class ResidualFoldEnv(gym.Env):
    """
    Residual RL Environment for garment folding.
    
    The RL agent learns RESIDUALS (corrections) to add to a frozen IL policy.
    This approach is easier than learning from scratch because:
    1. IL policy provides a strong baseline
    2. RL only needs to learn small corrections
    3. Bounded residuals ensure stability
    
    Action Space (Continuous):
        Residual delta for end-effector positions (8D):
        [left_dx, left_dy, left_dz, right_dx, right_dy, right_dz, left_grip_delta, right_grip_delta]
        All bounded to [-residual_scale, +residual_scale]
        
    Observation Space (Dict):
        - garment_pcd: (2048, 3) garment point cloud
        - joint_positions: (60,) current joint positions
        - ee_poses: (14,) end-effector positions + orientations
        - il_action: (8,) the action IL policy would take
        - gam_keypoints: (6, 3) manipulation keypoints
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        il_policy_type: str = "dummy",  # "SADP", "DP3", or "dummy" for testing
        task_name: str = "Fold_Tops",
        checkpoint_num: int = 1000,
        data_num: int = 100,
        config: Optional[Dict] = None,
        render_mode: str = "human",
        max_episode_steps: int = 300,
        residual_scale: float = 0.1,  # Max magnitude of residual
        action_scale: float = 0.05,   # Scale for converting to robot commands
        point_cloud_size: int = 2048,
        device: str = "cuda:0",
    ):
        """
        Initialize Residual RL environment.
        
        Args:
            il_policy_type: Type of IL policy to use
            task_name: Task name for IL policy checkpoint
            checkpoint_num: IL policy checkpoint number
            data_num: IL policy data configuration
            config: Additional environment config
            render_mode: Rendering mode
            max_episode_steps: Max steps per episode
            residual_scale: Maximum residual magnitude (bounds RL actions)
            action_scale: Scale for robot commands
            point_cloud_size: Number of points in observation
            device: Device for IL policy inference
        """
        super().__init__()
        
        self.il_policy_type = il_policy_type
        self.task_name = task_name
        self.checkpoint_num = checkpoint_num
        self.data_num = data_num
        self.config = config or {}
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.residual_scale = residual_scale
        self.action_scale = action_scale
        self.point_cloud_size = point_cloud_size
        self.device = device
        
        # Will be initialized lazily
        self._initialized = False
        self._il_policy = None
        
        # Episode state
        self.current_step = 0
        self._initial_bbox = None
        self._last_il_action = None
        
        # Action space: bounded residuals
        # 8D: [left_dxyz(3), right_dxyz(3), left_grip(1), right_grip(1)]
        self.action_space = spaces.Box(
            low=-residual_scale,
            high=residual_scale,
            shape=(8,),
            dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Dict({
            "garment_pcd": spaces.Box(
                low=-10.0, high=10.0,
                shape=(point_cloud_size, 3),
                dtype=np.float32
            ),
            "joint_positions": spaces.Box(
                low=-2 * np.pi, high=2 * np.pi,
                shape=(60,),
                dtype=np.float32
            ),
            "ee_poses": spaces.Box(
                low=-10.0, high=10.0,
                shape=(14,),
                dtype=np.float32
            ),
            "il_action": spaces.Box(
                low=-1.0, high=1.0,
                shape=(8,),
                dtype=np.float32
            ),
            "gam_keypoints": spaces.Box(
                low=-10.0, high=10.0,
                shape=(6, 3),
                dtype=np.float32
            ),
        })
        
        # Reward weights
        self.reward_weights = {
            "fold_progress": 1.0,
            "compactness": 0.5,
            "height_penalty": 0.3,
            "residual_penalty": 0.05,  # Penalize large residuals
            "success_bonus": 10.0,
        }
        
    def _lazy_init(self):
        """Initialize Isaac Sim environment and IL policy."""
        if self._initialized:
            return
        
        # Import Isaac Sim modules
        from Env_StandAlone.BaseEnv import BaseEnv
        from Env_Config.Garment.Particle_Garment import Particle_Garment
        from Env_Config.Robot.BimanualDex_Ur10e import Bimanual_Ur10e
        from Env_Config.Camera.Recording_Camera import Recording_Camera
        from Env_Config.Room.Real_Ground import Real_Ground
        from Model_HALO.GAM.GAM_Encapsulation import GAM_Encapsulation
        from Env_Config.Room.Object_Tools import set_prim_visible_group
        
        self._set_prim_visible_group = set_prim_visible_group
        
        print("[ResidualFoldEnv] Initializing environment...")
        
        # Create base environment
        self._base_env = BaseEnv()
        
        # Add ground
        self._ground = Real_Ground(
            self._base_env.scene,
            visual_material_usd=self.config.get("ground_material_usd"),
        )
        
        # Add garment - MUST use Tops garment for Fold_Tops task!
        garment_usd = self.config.get(
            "garment_usd_path",
            os.getcwd() + "/Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_018/TCLC_018_obj.usd"
        )
        self._garment = Particle_Garment(
            self._base_env.world,
            pos=np.array([0, 3.0, 0.6]),
            ori=np.array([0.0, 0.0, 0.0]),
            usd_path=garment_usd,
            contact_offset=0.012,
            rest_offset=0.010,
            particle_contact_offset=0.012,
            fluid_rest_offset=0.010,
            solid_rest_offset=0.010,
        )
        print(f"[ResidualFoldEnv] Using garment: {garment_usd}")
        
        # Add robot
        self._robot = Bimanual_Ur10e(
            self._base_env.world,
            dexleft_pos=np.array([-0.8, 0.0, 0.5]),
            dexleft_ori=np.array([0.0, 0.0, 0.0]),
            dexright_pos=np.array([0.8, 0.0, 0.5]),
            dexright_ori=np.array([0.0, 0.0, 0.0]),
        )
        
        # Add cameras
        self._garment_camera = Recording_Camera(
            camera_position=np.array([0.0, 1.0, 6.75]),
            camera_orientation=np.array([0, 90.0, 90.0]),
            prim_path="/World/garment_camera",
        )
        
        self._env_camera = Recording_Camera(
            camera_position=np.array([0.0, 4.0, 6.0]),
            camera_orientation=np.array([0, 60, -90.0]),
            prim_path="/World/env_camera",
        )
        
        # Load GAM model
        self._gam_model = GAM_Encapsulation(catogory="Tops_LongSleeve")
        
        # Initialize world
        self._base_env.reset()
        
        # Initialize cameras
        self._garment_camera.initialize(
            segment_pc_enable=True,
            segment_prim_path_list=["/World/Garment/garment"]
        )
        self._env_camera.initialize(depth_enable=True)
        
        # Open hands
        self._robot.set_both_hand_state("open", "open")
        
        # Step to settle
        for _ in range(100):
            self._base_env.step()
        
        # Load IL policy
        print(f"[ResidualFoldEnv] Loading IL policy ({self.il_policy_type})...")
        self._il_policy = create_il_policy(
            policy_type=self.il_policy_type,
            task_name=self.task_name,
            checkpoint_num=self.checkpoint_num,
            data_num=self.data_num,
            device=self.device,
        )
        if self.il_policy_type != "dummy":
            self._il_policy.load()
        
        self._initialized = True
        print("[ResidualFoldEnv] Environment initialized!")
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset environment."""
        super().reset(seed=seed)
        
        self._lazy_init()
        
        self.current_step = 0
        self._il_policy.reset()
        
        # Randomize garment position
        if seed is not None:
            np.random.seed(seed)
        
        pos_x = np.random.uniform(-0.1, 0.1)
        pos_y = np.random.uniform(0.7, 0.9)
        pos = np.array([pos_x, pos_y, 0.2])
        
        self._garment.set_pose(pos=pos, ori=np.array([0.0, 0.0, 0.0]))
        
        # Reset robot
        self._robot.dexleft.post_reset()
        self._robot.dexright.post_reset()
        self._robot.set_both_hand_state("open", "open")
        
        # Settle
        for _ in range(100):
            self._base_env.step()
        
        # Get observation
        obs = self._get_observation()
        
        # Store initial state
        self._initial_bbox = self._compute_bbox(obs["garment_pcd"])
        
        return obs, {"initial_pos": pos}
    
    def step(self, residual_action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute step with residual action.
        
        Args:
            residual_action: Residual to add to IL action (8D, bounded)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        # Get IL policy's action
        il_obs = self._get_il_observation()
        il_action = self._get_il_action(il_obs)
        
        # Combine: final = IL + residual
        # Clip residual to ensure bounds
        residual_action = np.clip(residual_action, -self.residual_scale, self.residual_scale)
        final_action = il_action + residual_action
        
        # Parse combined action
        left_delta = final_action[0:3] * self.action_scale
        right_delta = final_action[3:6] * self.action_scale
        left_grip = final_action[6] > 0.0
        right_grip = final_action[7] > 0.0
        
        # Get current EE positions
        left_pos, left_ori = self._robot.dexleft.get_cur_ee_pos()
        right_pos, right_ori = self._robot.dexright.get_cur_ee_pos()
        
        # Compute targets
        left_target = np.clip(left_pos + left_delta, [-1.5, -0.5, 0.0], [0.5, 2.0, 1.5])
        right_target = np.clip(right_pos + right_delta, [-0.5, -0.5, 0.0], [1.5, 2.0, 1.5])
        
        # Execute robot movements
        try:
            self._robot.dexleft.step_action(
                target_pos=left_target,
                target_ori=np.array([0.579, -0.579, -0.406, 0.406]),
                angular_type="quat"
            )
        except:
            pass
        
        try:
            self._robot.dexright.step_action(
                target_pos=right_target,
                target_ori=np.array([0.406, -0.406, -0.579, 0.579]),
                angular_type="quat"
            )
        except:
            pass
        
        # Apply gripper
        self._robot.set_both_hand_state(
            "close" if left_grip else "open",
            "close" if right_grip else "open"
        )
        
        # Step physics
        for _ in range(5):
            self._base_env.step()
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward
        reward, reward_info = self._compute_reward(obs, residual_action)
        
        # Check termination
        terminated = self._check_success(obs)
        truncated = self.current_step >= self.max_episode_steps
        
        info = {
            "step": self.current_step,
            "il_action": il_action,
            "residual": residual_action,
            "final_action": final_action,
            "is_success": terminated,
            **reward_info,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get observation including IL action."""
        
        # Get garment point cloud
        garment_pcd = self._get_garment_pcd()
        
        # Get joint positions
        left_joints = self._robot.dexleft.get_joint_positions()
        right_joints = self._robot.dexright.get_joint_positions()
        joint_positions = np.concatenate([left_joints, right_joints]).astype(np.float32)
        
        # Get EE poses
        left_pos, left_ori = self._robot.dexleft.get_cur_ee_pos()
        right_pos, right_ori = self._robot.dexright.get_cur_ee_pos()
        ee_poses = np.concatenate([left_pos, left_ori, right_pos, right_ori]).astype(np.float32)
        
        # Get GAM keypoints
        try:
            keypoints, _, _ = self._gam_model.get_manipulation_points(
                input_pcd=garment_pcd,
                index_list=[957, 501, 1902, 448, 1196, 422]
            )
            gam_keypoints = keypoints.astype(np.float32)
        except:
            gam_keypoints = np.zeros((6, 3), dtype=np.float32)
        
        # Get IL action for this observation
        il_obs = self._prepare_il_obs(garment_pcd, joint_positions, gam_keypoints)
        il_action = self._get_il_action(il_obs)
        self._last_il_action = il_action
        
        return {
            "garment_pcd": garment_pcd.astype(np.float32),
            "joint_positions": joint_positions,
            "ee_poses": ee_poses,
            "il_action": il_action.astype(np.float32),
            "gam_keypoints": gam_keypoints,
        }
    
    def _get_il_observation(self) -> Dict[str, np.ndarray]:
        """Get observation in format expected by IL policy."""
        garment_pcd = self._get_garment_pcd()
        left_joints = self._robot.dexleft.get_joint_positions()
        right_joints = self._robot.dexright.get_joint_positions()
        joint_positions = np.concatenate([left_joints, right_joints])
        
        try:
            keypoints, _, similarity = self._gam_model.get_manipulation_points(
                input_pcd=garment_pcd,
                index_list=[957, 501, 1902, 448, 1196, 422]
            )
            # Create affordance features from similarity
            affordance = np.zeros((self.point_cloud_size, 2), dtype=np.float32)
            if similarity is not None and len(similarity) >= 2:
                affordance[:, 0] = similarity[0][:self.point_cloud_size] if len(similarity[0]) >= self.point_cloud_size else 0
                affordance[:, 1] = similarity[1][:self.point_cloud_size] if len(similarity[1]) >= self.point_cloud_size else 0
        except:
            affordance = np.zeros((self.point_cloud_size, 2), dtype=np.float32)
        
        return self._prepare_il_obs(garment_pcd, joint_positions, affordance)
    
    def _prepare_il_obs(
        self,
        garment_pcd: np.ndarray,
        joint_positions: np.ndarray,
        affordance_or_keypoints: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Prepare observation for IL policy."""
        # Get environment point cloud
        env_pcd = self._env_camera.get_pointcloud_from_depth(
            show_original_pc_online=False,
            show_downsample_pc_online=False,
        ) if hasattr(self, '_env_camera') else garment_pcd
        
        return {
            "environment_point_cloud": env_pcd,
            "garment_point_cloud": garment_pcd,
            "object_point_cloud": garment_pcd,  # Same as garment for folding
            "points_affordance_feature": affordance_or_keypoints if affordance_or_keypoints.shape[-1] == 2 
                else np.zeros((self.point_cloud_size, 2), dtype=np.float32),
            "agent_pos": joint_positions,
        }
    
    def _get_il_action(self, il_obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Get action from IL policy and convert to 8D format."""
        # Get raw IL action (may be high-dimensional joint action)
        raw_action = self._il_policy.get_action_single_step(il_obs)
        
        # For dummy policy or if action is already 8D
        if len(raw_action) == 8:
            return raw_action
        
        # Convert joint-space action to EE-space action
        # This is a simplification - in practice, you'd use the actual mapping
        # For now, extract first 8 values or use default
        if len(raw_action) >= 8:
            return raw_action[:8]
        else:
            return np.zeros(8, dtype=np.float32)
    
    def _get_garment_pcd(self) -> np.ndarray:
        """Get garment point cloud."""
        self._set_prim_visible_group(
            prim_path_list=["/World/DexLeft", "/World/DexRight"],
            visible=False,
        )
        for _ in range(2):
            self._base_env.step()
        
        pcd, _ = self._garment_camera.get_point_cloud_data_from_segment(
            save_or_not=False,
            real_time_watch=False,
        )
        
        self._set_prim_visible_group(
            prim_path_list=["/World/DexLeft", "/World/DexRight"],
            visible=True,
        )
        for _ in range(2):
            self._base_env.step()
        
        return self._normalize_pcd(pcd)
    
    def _normalize_pcd(self, pcd: np.ndarray) -> np.ndarray:
        """Normalize point cloud size."""
        n = len(pcd)
        if n == 0:
            return np.zeros((self.point_cloud_size, 3), dtype=np.float32)
        if n >= self.point_cloud_size:
            idx = np.random.choice(n, self.point_cloud_size, replace=False)
        else:
            idx = np.random.choice(n, self.point_cloud_size, replace=True)
        return pcd[idx]
    
    def _compute_bbox(self, pcd: np.ndarray) -> np.ndarray:
        """Compute bounding box."""
        if len(pcd) == 0:
            return np.zeros(6)
        return np.concatenate([np.min(pcd, axis=0), np.max(pcd, axis=0)])
    
    def _compute_reward(
        self,
        obs: Dict[str, np.ndarray],
        residual: np.ndarray,
    ) -> Tuple[float, Dict]:
        """Compute reward with residual penalty."""
        reward = 0.0
        info = {}
        
        pcd = obs["garment_pcd"]
        current_bbox = self._compute_bbox(pcd)
        current_size = current_bbox[3:6] - current_bbox[0:3]
        initial_size = self._initial_bbox[3:6] - self._initial_bbox[0:3]
        
        # Fold progress
        initial_xy = initial_size[0] * initial_size[1]
        current_xy = current_size[0] * current_size[1]
        fold_progress = (initial_xy - current_xy) / (initial_xy + 1e-6)
        reward += fold_progress * self.reward_weights["fold_progress"]
        info["fold_progress"] = fold_progress
        
        # Compactness
        initial_vol = np.prod(initial_size)
        current_vol = np.prod(current_size)
        compactness = 1.0 - current_vol / (initial_vol + 1e-6)
        reward += compactness * self.reward_weights["compactness"]
        info["compactness"] = compactness
        
        # Height penalty
        height_var = np.var(pcd[:, 2])
        reward -= height_var * self.reward_weights["height_penalty"]
        info["height_penalty"] = -height_var
        
        # Residual penalty (encourage small corrections)
        residual_magnitude = np.sum(residual ** 2)
        reward -= residual_magnitude * self.reward_weights["residual_penalty"]
        info["residual_penalty"] = -residual_magnitude
        
        # Success bonus
        if self._check_success(obs):
            reward += self.reward_weights["success_bonus"]
            info["success_bonus"] = self.reward_weights["success_bonus"]
        
        info["total_reward"] = reward
        return reward, info
    
    def _check_success(self, obs: Dict[str, np.ndarray]) -> bool:
        """Check if folding is complete."""
        pcd = obs["garment_pcd"]
        current_bbox = self._compute_bbox(pcd)
        current_size = current_bbox[3:6] - current_bbox[0:3]
        initial_size = self._initial_bbox[3:6] - self._initial_bbox[0:3]
        
        x_ratio = current_size[0] / (initial_size[0] + 1e-6)
        y_ratio = current_size[1] / (initial_size[1] + 1e-6)
        height_var = np.var(pcd[:, 2])
        
        return (x_ratio < 0.5) and (y_ratio < 0.7) and (height_var < 0.02)
    
    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self._env_camera.get_rgb_graph(save_or_not=False)
        return None
    
    def close(self):
        if self._initialized:
            self._base_env.stop()
            print("[ResidualFoldEnv] Environment closed.")


def register_residual_env():
    """Register environment."""
    gym.register(
        id="ResidualFoldTops-v0",
        entry_point="Env_RL.residual_fold_env:ResidualFoldEnv",
        max_episode_steps=300,
    )



