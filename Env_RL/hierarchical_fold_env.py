"""
Hierarchical RL Environment for Garment Folding.

This environment implements a hierarchical approach where:
- HIGH-LEVEL: RL policy selects which primitive to execute (discrete action space)
- LOW-LEVEL: Pre-defined primitives handle actual robot control

The high-level policy learns the optimal SEQUENCE of primitives, while
the low-level primitives handle the motor control using existing DexGarmentLab code.

This dramatically reduces the RL action space from 60+ continuous DOF to
6 discrete primitive choices, making learning much faster.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Env_RL.primitives import ManipulationPrimitives, PrimitiveID, PrimitiveResult


class HierarchicalFoldEnv(gym.Env):
    """
    Hierarchical Gymnasium environment for garment folding.
    
    The agent selects HIGH-LEVEL primitives (fold_left, fold_right, etc.)
    and the environment executes them using pre-defined LOW-LEVEL controllers.
    
    Action Space (Discrete):
        0: FOLD_LEFT_SLEEVE  - Fold the left sleeve inward
        1: FOLD_RIGHT_SLEEVE - Fold the right sleeve inward  
        2: FOLD_BOTTOM       - Fold bottom half up
        3: OPEN_HANDS        - Open both grippers
        4: MOVE_TO_HOME      - Move arms to home position
        5: DONE              - Signal task completion
        
    Observation Space (Dict):
        - garment_pcd: Point cloud of garment (N, 3)
        - gam_keypoints: GAM-detected manipulation points (6, 3)
        - primitive_mask: Which primitives are still available (6,)
        - executed_sequence: One-hot of executed primitives (6,)
        
    Reward:
        - +3.0 for each successful primitive
        - +10.0 bonus for completing all three folds
        - -1.0 for failed primitive
        - -0.5 for selecting already-executed primitive
        - -2.0 for selecting DONE before task is complete
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        render_mode: str = "human",
        max_primitives: int = 10,  # Max primitive selections per episode
        point_cloud_size: int = 2048,
    ):
        """
        Initialize hierarchical environment.
        
        Args:
            config: Configuration dict (usd_path, ground_material_usd)
            render_mode: "human" or "rgb_array"
            max_primitives: Maximum number of primitive selections per episode
            point_cloud_size: Number of points in observation point cloud
        """
        super().__init__()
        
        self.config = config or {}
        self.render_mode = render_mode
        self.max_primitives = max_primitives
        self.point_cloud_size = point_cloud_size
        
        # Will be initialized lazily
        self._initialized = False
        self._primitives = None
        
        # Episode state
        self.current_step = 0
        self.total_physics_steps = 0
        self.executed_sequence = []
        
        # Define action space: 6 discrete primitive choices
        self.action_space = spaces.Discrete(len(PrimitiveID))
        
        # Define observation space
        self.observation_space = spaces.Dict({
            "garment_pcd": spaces.Box(
                low=-10.0, high=10.0,
                shape=(self.point_cloud_size, 3),
                dtype=np.float32
            ),
            "gam_keypoints": spaces.Box(
                low=-10.0, high=10.0,
                shape=(6, 3),
                dtype=np.float32
            ),
            "primitive_mask": spaces.Box(
                low=0, high=1,
                shape=(len(PrimitiveID),),
                dtype=np.float32
            ),
            "executed_sequence": spaces.Box(
                low=0, high=1,
                shape=(len(PrimitiveID),),
                dtype=np.float32
            ),
        })
        
        # Track initial garment state for reward computation
        self._initial_bbox = None
        
    def _lazy_init(self):
        """Initialize Isaac Sim environment lazily."""
        if self._initialized:
            return
            
        # Import Isaac Sim modules (SimulationApp must be created before this)
        from Env_StandAlone.BaseEnv import BaseEnv
        from Env_Config.Garment.Particle_Garment import Particle_Garment
        from Env_Config.Robot.BimanualDex_Ur10e import Bimanual_Ur10e
        from Env_Config.Camera.Recording_Camera import Recording_Camera
        from Env_Config.Room.Real_Ground import Real_Ground
        from Model_HALO.GAM.GAM_Encapsulation import GAM_Encapsulation
        from Env_Config.Room.Object_Tools import set_prim_visible_group
        
        self._set_prim_visible_group = set_prim_visible_group
        
        print("[HierarchicalFoldEnv] Initializing Isaac Sim environment...")
        
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
        print(f"[HierarchicalFoldEnv] Using garment: {garment_usd}")
        
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
        
        # Open hands initially
        self._robot.set_both_hand_state("open", "open")
        
        # Step to settle
        for _ in range(100):
            self._base_env.step()
        
        # Create manipulation primitives
        self._primitives = ManipulationPrimitives(
            robot=self._robot,
            garment=self._garment,
            gam_model=self._gam_model,
            base_env=self._base_env,
            garment_camera=self._garment_camera,
        )
        
        self._initialized = True
        print("[HierarchicalFoldEnv] Environment initialized successfully!")
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation: Initial observation
            info: Additional info
        """
        super().reset(seed=seed)
        
        # Initialize if needed
        self._lazy_init()
        
        # Reset episode state
        self.current_step = 0
        self.total_physics_steps = 0
        self.executed_sequence = []
        self._primitives.reset()
        
        # Randomize garment position
        if seed is not None:
            np.random.seed(seed)
        
        pos_x = np.random.uniform(-0.1, 0.1)
        pos_y = np.random.uniform(0.7, 0.9)
        pos = np.array([pos_x, pos_y, 0.2])
        ori = np.array([0.0, 0.0, 0.0])
        
        # Reset garment
        self._garment.set_pose(pos=pos, ori=ori)
        
        # Reset robot
        self._robot.dexleft.post_reset()
        self._robot.dexright.post_reset()
        self._robot.set_both_hand_state("open", "open")
        
        # Step to settle
        for _ in range(100):
            self._base_env.step()
        
        # Get initial observation
        obs = self._get_observation()
        
        # Store initial bbox for reward computation
        self._initial_bbox = self._compute_bbox(obs["garment_pcd"])
        
        info = {
            "initial_pos": pos,
            "garment_usd": self._garment.usd_path,
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute a high-level primitive selection.
        
        Args:
            action: Primitive ID to execute (0-5)
            
        Returns:
            observation: New observation after primitive execution
            reward: Reward for this primitive selection
            terminated: Whether task completed successfully
            truncated: Whether episode was cut short
            info: Additional information
        """
        self.current_step += 1
        primitive_id = PrimitiveID(action)
        
        # Get current garment point cloud
        garment_pcd = self._get_garment_pcd()
        
        # Check if primitive was already executed (except DONE and utility primitives)
        already_executed = primitive_id in self._primitives.get_executed_primitives()
        is_folding_primitive = primitive_id in {
            PrimitiveID.FOLD_LEFT_SLEEVE,
            PrimitiveID.FOLD_RIGHT_SLEEVE,
            PrimitiveID.FOLD_BOTTOM
        }
        
        # Initialize reward and info
        reward = 0.0
        info = {
            "step": self.current_step,
            "primitive": primitive_id.name,
            "primitive_id": int(primitive_id),
        }
        
        # Handle DONE action
        if primitive_id == PrimitiveID.DONE:
            if self._primitives.is_complete_sequence():
                # Task completed successfully!
                reward = 10.0  # Big bonus for completing the task
                info["success"] = True
                info["message"] = "Task completed successfully!"
                obs = self._get_observation()
                return obs, reward, True, False, info
            else:
                # Premature DONE - penalty
                reward = -2.0
                info["success"] = False
                info["message"] = "DONE selected but task not complete"
                obs = self._get_observation()
                return obs, reward, False, False, info
        
        # Penalty for re-selecting already executed folding primitive
        if already_executed and is_folding_primitive:
            reward = -0.5
            info["message"] = f"Primitive {primitive_id.name} already executed"
            obs = self._get_observation()
            return obs, reward, False, False, info
        
        # Execute the primitive
        result = self._primitives.execute(
            primitive_id=primitive_id,
            garment_pcd=garment_pcd,
            set_prim_visible_group_func=self._set_prim_visible_group,
        )
        
        self.total_physics_steps += result.steps_taken
        info.update(result.info)
        info["physics_steps"] = result.steps_taken
        
        # Compute reward based on primitive result
        if result.success:
            if is_folding_primitive:
                reward = 3.0  # Reward for successful folding primitive
                
                # Bonus for good fold quality
                obs = self._get_observation()
                fold_quality = self._compute_fold_quality(obs["garment_pcd"])
                reward += fold_quality * 2.0  # Up to +2 bonus
                
                info["fold_quality"] = fold_quality
            else:
                reward = 0.1  # Small reward for utility primitives
        else:
            reward = -1.0  # Penalty for failed primitive
            info["error"] = result.info.get("error", "Unknown error")
        
        # Check if all primitives done (but agent hasn't said DONE yet)
        if self._primitives.is_complete_sequence():
            reward += 5.0  # Bonus for completing all folds
            info["all_folds_complete"] = True
        
        # Track sequence
        self.executed_sequence.append(primitive_id)
        
        # Get new observation
        obs = self._get_observation()
        
        # Check termination conditions
        terminated = False  # Only terminate when agent says DONE
        truncated = self.current_step >= self.max_primitives
        
        if truncated:
            info["message"] = "Episode truncated (max primitives reached)"
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        
        # Get garment point cloud
        garment_pcd = self._get_garment_pcd()
        
        # Get GAM keypoints
        try:
            keypoints, _, _ = self._gam_model.get_manipulation_points(
                input_pcd=garment_pcd,
                index_list=[957, 501, 1902, 448, 1196, 422]
            )
            gam_keypoints = keypoints.astype(np.float32)
        except Exception:
            gam_keypoints = np.zeros((6, 3), dtype=np.float32)
        
        # Build primitive mask (which primitives are available)
        executed = self._primitives.get_executed_primitives()
        primitive_mask = np.ones(len(PrimitiveID), dtype=np.float32)
        for pid in executed:
            if pid in {PrimitiveID.FOLD_LEFT_SLEEVE, PrimitiveID.FOLD_RIGHT_SLEEVE, PrimitiveID.FOLD_BOTTOM}:
                primitive_mask[pid] = 0.0  # Folding primitives can only be done once
        
        # Build executed sequence (one-hot)
        executed_sequence = np.zeros(len(PrimitiveID), dtype=np.float32)
        for pid in executed:
            executed_sequence[pid] = 1.0
        
        return {
            "garment_pcd": garment_pcd.astype(np.float32),
            "gam_keypoints": gam_keypoints,
            "primitive_mask": primitive_mask,
            "executed_sequence": executed_sequence,
        }
    
    def _get_garment_pcd(self) -> np.ndarray:
        """Get garment point cloud, hiding robot temporarily."""
        
        # Hide robots
        self._set_prim_visible_group(
            prim_path_list=["/World/DexLeft", "/World/DexRight"],
            visible=False,
        )
        for _ in range(2):
            self._base_env.step()
        
        # Get point cloud
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
        
        # Normalize to fixed size
        pcd = self._normalize_point_cloud(pcd)
        
        return pcd
    
    def _normalize_point_cloud(self, pcd: np.ndarray) -> np.ndarray:
        """Normalize point cloud to fixed size."""
        n_points = len(pcd)
        
        if n_points == 0:
            return np.zeros((self.point_cloud_size, 3), dtype=np.float32)
        
        if n_points >= self.point_cloud_size:
            indices = np.random.choice(n_points, self.point_cloud_size, replace=False)
            return pcd[indices]
        else:
            indices = np.random.choice(n_points, self.point_cloud_size, replace=True)
            return pcd[indices]
    
    def _compute_bbox(self, pcd: np.ndarray) -> np.ndarray:
        """Compute bounding box."""
        if len(pcd) == 0:
            return np.zeros(6)
        return np.concatenate([np.min(pcd, axis=0), np.max(pcd, axis=0)])
    
    def _compute_fold_quality(self, pcd: np.ndarray) -> float:
        """
        Compute quality of current fold (0 to 1).
        
        Based on reduction in bounding box area compared to initial.
        """
        if self._initial_bbox is None:
            return 0.0
        
        current_bbox = self._compute_bbox(pcd)
        
        # Compute XY areas
        initial_area = (self._initial_bbox[3] - self._initial_bbox[0]) * \
                       (self._initial_bbox[4] - self._initial_bbox[1])
        current_area = (current_bbox[3] - current_bbox[0]) * \
                       (current_bbox[4] - current_bbox[1])
        
        if initial_area < 1e-6:
            return 0.0
        
        # Fold quality is how much area was reduced
        reduction = (initial_area - current_area) / initial_area
        return np.clip(reduction, 0.0, 1.0)
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._env_camera.get_rgb_graph(save_or_not=False)
        return None
    
    def close(self):
        """Clean up."""
        if self._initialized:
            self._base_env.stop()
            print("[HierarchicalFoldEnv] Environment closed.")


def register_hierarchical_env():
    """Register environment with Gymnasium."""
    gym.register(
        id="HierarchicalFoldTops-v0",
        entry_point="Env_RL.hierarchical_fold_env:HierarchicalFoldEnv",
        max_episode_steps=10,
    )



