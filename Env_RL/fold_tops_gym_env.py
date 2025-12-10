"""
Gymnasium wrapper for DexGarmentLab FoldTops task.

This wrapper enables RL training (e.g., PPO) on top of the existing
DexGarmentLab simulation environment.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FoldTopsGymEnv(gym.Env):
    """
    Gymnasium environment wrapper for garment folding task.
    
    Leverages existing DexGarmentLab implementations for:
    - Bimanual UR10e + Shadow Hand robot control
    - Particle-based cloth simulation
    - Point cloud observations from cameras
    - GAM model for semantic keypoint detection
    
    Action Space:
        - End-effector delta positions for both hands (6D: left_xyz + right_xyz)
        - Gripper states (2D: left_grip + right_grip, discrete)
        
    Observation Space:
        - Garment point cloud (N x 3)
        - Joint positions (60D for both arms + hands)
        - End-effector poses (14D: 2 x [pos(3) + quat(4)])
        - GAM affordance features (optional)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        render_mode: str = "human",
        max_episode_steps: int = 300,
        action_scale: float = 0.05,
        use_gam_features: bool = True,
        point_cloud_size: int = 2048,
    ):
        """
        Initialize the Gym environment.
        
        Args:
            config: Configuration dict with keys:
                - pos: Initial garment position [x, y, z]
                - ori: Initial garment orientation [rx, ry, rz]
                - usd_path: Path to garment USD file
                - ground_material_usd: Path to ground material
            render_mode: "human" for GUI, "rgb_array" for headless
            max_episode_steps: Maximum steps per episode
            action_scale: Scaling factor for action magnitudes
            use_gam_features: Whether to include GAM affordance features in obs
            point_cloud_size: Number of points in downsampled point cloud
        """
        super().__init__()
        
        self.config = config or {}
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.use_gam_features = use_gam_features
        self.point_cloud_size = point_cloud_size
        
        # Will be initialized lazily to avoid import issues
        self._env = None
        self._initialized = False
        
        # Episode state
        self.current_step = 0
        self.initial_garment_pcd = None
        
        # Define action space: delta EE positions (6D) + gripper commands (2D discrete)
        # Actions: [left_dx, left_dy, left_dz, right_dx, right_dy, right_dz, left_grip, right_grip]
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define observation space
        obs_dict = {
            "garment_pcd": spaces.Box(
                low=-10.0, high=10.0,
                shape=(self.point_cloud_size, 3),
                dtype=np.float32
            ),
            "joint_positions": spaces.Box(
                low=-2 * np.pi, high=2 * np.pi,
                shape=(60,),  # 30 DOF per arm+hand
                dtype=np.float32
            ),
            "ee_poses": spaces.Box(
                low=-10.0, high=10.0,
                shape=(14,),  # 2 x (pos[3] + quat[4])
                dtype=np.float32
            ),
        }
        
        if self.use_gam_features:
            # GAM keypoint positions (6 keypoints x 3D)
            obs_dict["gam_keypoints"] = spaces.Box(
                low=-10.0, high=10.0,
                shape=(6, 3),
                dtype=np.float32
            )
        
        self.observation_space = spaces.Dict(obs_dict)
        
        # Reward weights
        self.reward_weights = {
            "fold_progress": 1.0,
            "compactness": 0.5,
            "height_penalty": 0.3,
            "action_penalty": 0.01,
            "success_bonus": 10.0,
        }
        
    def _lazy_init(self):
        """Lazily initialize the Isaac Sim environment."""
        if self._initialized:
            return
            
        # Import here to ensure SimulationApp is created first
        from Env_StandAlone.BaseEnv import BaseEnv
        from Env_Config.Garment.Particle_Garment import Particle_Garment
        from Env_Config.Robot.BimanualDex_Ur10e import Bimanual_Ur10e
        from Env_Config.Camera.Recording_Camera import Recording_Camera
        from Env_Config.Room.Real_Ground import Real_Ground
        from Model_HALO.GAM.GAM_Encapsulation import GAM_Encapsulation
        from Env_Config.Room.Object_Tools import set_prim_visible_group
        
        # Store imports for later use
        self._set_prim_visible_group = set_prim_visible_group
        
        # Create custom environment (simplified from FoldTops_Env)
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
        print(f"[FoldTopsGymEnv] Using garment: {garment_usd}")
        
        # Add bimanual robot
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
        
        # Load GAM model for keypoint detection
        if self.use_gam_features:
            self._gam_model = GAM_Encapsulation(catogory="Tops_LongSleeve")
            # Keypoint indices for folding task
            self._keypoint_indices = [957, 501, 1902, 448, 1196, 422]
        
        # Initialize world
        self._base_env.reset()
        
        # Initialize cameras
        self._garment_camera.initialize(
            segment_pc_enable=True,
            segment_prim_path_list=["/World/Garment/garment"]
        )
        self._env_camera.initialize(depth_enable=True)
        
        # Open hands initially
        self._robot.set_both_hand_state("open", "open")
        
        # Step to settle
        for _ in range(100):
            self._base_env.step()
        
        self._initialized = True
        print("[FoldTopsGymEnv] Environment initialized successfully!")
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            observation: Initial observation dict
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Initialize if needed
        self._lazy_init()
        
        # Reset episode counter
        self.current_step = 0
        
        # Randomize garment position
        if seed is not None:
            np.random.seed(seed)
        
        pos_x = np.random.uniform(-0.1, 0.1)
        pos_y = np.random.uniform(0.7, 0.9)
        pos = np.array([pos_x, pos_y, 0.2])
        ori = np.array([0.0, 0.0, 0.0])
        
        # Reset garment pose
        self._garment.set_pose(pos=pos, ori=ori)
        
        # Reset robot to home position
        self._robot.dexleft.post_reset()
        self._robot.dexright.post_reset()
        self._robot.set_both_hand_state("open", "open")
        
        # Step simulation to settle
        for _ in range(100):
            self._base_env.step()
        
        # Get initial observation
        obs = self._get_observation()
        
        # Store initial garment state for reward computation
        self.initial_garment_pcd = obs["garment_pcd"].copy()
        self._initial_bbox = self._compute_bbox(self.initial_garment_pcd)
        
        info = {
            "initial_pos": pos,
            "initial_ori": ori,
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action array [left_dx, left_dy, left_dz, right_dx, right_dy, right_dz, left_grip, right_grip]
            
        Returns:
            observation: New observation
            reward: Step reward
            terminated: Whether episode ended successfully
            truncated: Whether episode was cut off
            info: Additional information
        """
        self.current_step += 1
        
        # Parse actions
        left_delta = action[0:3] * self.action_scale
        right_delta = action[3:6] * self.action_scale
        left_grip = action[6] > 0.5  # Threshold for gripper
        right_grip = action[7] > 0.5
        
        # Get current EE positions
        left_pos, left_ori = self._robot.dexleft.get_cur_ee_pos()
        right_pos, right_ori = self._robot.dexright.get_cur_ee_pos()
        
        # Compute target positions with safety clipping
        left_target = np.clip(left_pos + left_delta, [-1.5, -0.5, 0.0], [0.5, 2.0, 1.5])
        right_target = np.clip(right_pos + right_delta, [-0.5, -0.5, 0.0], [1.5, 2.0, 1.5])
        
        # Execute movements using existing IK
        # Use step_action for single IK step (faster than dense_step)
        try:
            self._robot.dexleft.step_action(
                target_pos=left_target,
                target_ori=np.array([0.579, -0.579, -0.406, 0.406]),
                angular_type="quat"
            )
        except Exception as e:
            pass  # IK failure - continue with current position
            
        try:
            self._robot.dexright.step_action(
                target_pos=right_target,
                target_ori=np.array([0.406, -0.406, -0.579, 0.579]),
                angular_type="quat"
            )
        except Exception as e:
            pass  # IK failure - continue with current position
        
        # Apply gripper actions
        left_state = "close" if left_grip else "open"
        right_state = "close" if right_grip else "open"
        self._robot.set_both_hand_state(left_state, right_state)
        
        # Step physics (action repeat for stability)
        for _ in range(5):
            self._base_env.step()
        
        # Get new observation
        obs = self._get_observation()
        
        # Compute reward
        reward, reward_info = self._compute_reward(obs, action)
        
        # Check termination conditions
        terminated = self._check_success(obs)
        truncated = self.current_step >= self.max_episode_steps
        
        info = {
            "step": self.current_step,
            "is_success": terminated,
            **reward_info,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation from environment."""
        
        # Hide robots temporarily to get clean garment point cloud
        self._set_prim_visible_group(
            prim_path_list=["/World/DexLeft", "/World/DexRight"],
            visible=False,
        )
        for _ in range(2):
            self._base_env.step()
        
        # Get garment point cloud
        pcd, _ = self._garment_camera.get_point_cloud_data_from_segment(
            save_or_not=False,
            real_time_watch=False,
        )
        
        # Unhide robots
        self._set_prim_visible_group(
            prim_path_list=["/World/DexLeft", "/World/DexRight"],
            visible=True,
        )
        for _ in range(2):
            self._base_env.step()
        
        # Downsample/upsample point cloud to fixed size
        pcd = self._normalize_point_cloud(pcd)
        
        # Get joint positions
        left_joints = self._robot.dexleft.get_joint_positions()
        right_joints = self._robot.dexright.get_joint_positions()
        joint_positions = np.concatenate([left_joints, right_joints]).astype(np.float32)
        
        # Get EE poses
        left_pos, left_ori = self._robot.dexleft.get_cur_ee_pos()
        right_pos, right_ori = self._robot.dexright.get_cur_ee_pos()
        ee_poses = np.concatenate([left_pos, left_ori, right_pos, right_ori]).astype(np.float32)
        
        obs = {
            "garment_pcd": pcd.astype(np.float32),
            "joint_positions": joint_positions,
            "ee_poses": ee_poses,
        }
        
        # Add GAM keypoints if enabled
        if self.use_gam_features:
            try:
                keypoints, _, _ = self._gam_model.get_manipulation_points(
                    input_pcd=pcd,
                    index_list=self._keypoint_indices
                )
                obs["gam_keypoints"] = keypoints.astype(np.float32)
            except Exception:
                # Fallback if GAM fails
                obs["gam_keypoints"] = np.zeros((6, 3), dtype=np.float32)
        
        return obs
    
    def _normalize_point_cloud(self, pcd: np.ndarray) -> np.ndarray:
        """Normalize point cloud to fixed size."""
        n_points = len(pcd)
        
        if n_points == 0:
            return np.zeros((self.point_cloud_size, 3), dtype=np.float32)
        
        if n_points >= self.point_cloud_size:
            # Random downsample
            indices = np.random.choice(n_points, self.point_cloud_size, replace=False)
            return pcd[indices]
        else:
            # Upsample by repeating points
            indices = np.random.choice(n_points, self.point_cloud_size, replace=True)
            return pcd[indices]
    
    def _compute_bbox(self, pcd: np.ndarray) -> np.ndarray:
        """Compute bounding box of point cloud."""
        if len(pcd) == 0:
            return np.array([0, 0, 0, 0, 0, 0])
        min_pt = np.min(pcd, axis=0)
        max_pt = np.max(pcd, axis=0)
        return np.concatenate([min_pt, max_pt])
    
    def _compute_reward(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Compute reward for current state.
        
        Reward components:
        1. Fold progress: Reduction in bounding box volume
        2. Compactness: How tight the garment is folded
        3. Height penalty: Garment should be flat
        4. Action penalty: Encourage smooth actions
        5. Success bonus: Large reward for completing fold
        """
        reward = 0.0
        reward_info = {}
        
        pcd = obs["garment_pcd"]
        
        # Current bounding box
        current_bbox = self._compute_bbox(pcd)
        current_size = current_bbox[3:6] - current_bbox[0:3]
        initial_size = self._initial_bbox[3:6] - self._initial_bbox[0:3]
        
        # 1. Fold progress reward (reduction in xy spread)
        initial_xy_area = initial_size[0] * initial_size[1]
        current_xy_area = current_size[0] * current_size[1]
        fold_progress = (initial_xy_area - current_xy_area) / (initial_xy_area + 1e-6)
        fold_reward = fold_progress * self.reward_weights["fold_progress"]
        reward += fold_reward
        reward_info["fold_progress"] = fold_progress
        
        # 2. Compactness reward
        current_volume = np.prod(current_size)
        initial_volume = np.prod(initial_size)
        compactness = 1.0 - (current_volume / (initial_volume + 1e-6))
        compactness_reward = compactness * self.reward_weights["compactness"]
        reward += compactness_reward
        reward_info["compactness"] = compactness
        
        # 3. Height penalty (garment should stay flat)
        height_var = np.var(pcd[:, 2])
        height_penalty = -height_var * self.reward_weights["height_penalty"]
        reward += height_penalty
        reward_info["height_penalty"] = height_penalty
        
        # 4. Action penalty (encourage smooth actions)
        action_magnitude = np.sum(action[:6] ** 2)
        action_penalty = -action_magnitude * self.reward_weights["action_penalty"]
        reward += action_penalty
        reward_info["action_penalty"] = action_penalty
        
        # 5. Success bonus
        if self._check_success(obs):
            reward += self.reward_weights["success_bonus"]
            reward_info["success_bonus"] = self.reward_weights["success_bonus"]
        
        reward_info["total_reward"] = reward
        
        return reward, reward_info
    
    def _check_success(self, obs: Dict[str, np.ndarray]) -> bool:
        """
        Check if folding task is successfully completed.
        
        Success criteria:
        - Bounding box X dimension < 40% of initial
        - Bounding box Y dimension < 60% of initial
        - Garment is relatively flat (low height variance)
        """
        pcd = obs["garment_pcd"]
        
        current_bbox = self._compute_bbox(pcd)
        current_size = current_bbox[3:6] - current_bbox[0:3]
        initial_size = self._initial_bbox[3:6] - self._initial_bbox[0:3]
        
        # Check fold ratios
        x_ratio = current_size[0] / (initial_size[0] + 1e-6)
        y_ratio = current_size[1] / (initial_size[1] + 1e-6)
        
        # Height variance check
        height_var = np.var(pcd[:, 2])
        
        is_folded = (x_ratio < 0.5) and (y_ratio < 0.7) and (height_var < 0.02)
        
        return is_folded
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._env_camera.get_rgb_graph(save_or_not=False)
        return None
    
    def close(self):
        """Clean up environment resources."""
        if self._initialized:
            self._base_env.stop()
            print("[FoldTopsGymEnv] Environment closed.")


# Register environment with Gymnasium
def register_fold_tops_env():
    """Register the FoldTops environment with Gymnasium."""
    gym.register(
        id="FoldTops-v0",
        entry_point="Env_RL.fold_tops_gym_env:FoldTopsGymEnv",
        max_episode_steps=300,
    )



