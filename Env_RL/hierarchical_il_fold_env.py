"""
Hierarchical RL Environment with IL (SADP_G) Primitives.

This environment implements a hierarchical approach where:
- HIGH-LEVEL: RL policy selects which stage/primitive to execute (discrete action space)
- LOW-LEVEL: Pre-trained SADP_G models handle actual robot control for each stage

Key Difference from hierarchical_fold_env.py:
- OLD: Uses hard-coded IK trajectories for primitives
- NEW: Uses trained SADP_G diffusion policies for each folding stage

The high-level RL policy learns the optimal SEQUENCE and TIMING of SADP_G stages,
while the trained IL models handle the actual manipulation.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Env_RL.il_primitives import ILManipulationPrimitives, ILPrimitiveID, ILPrimitiveResult
from Env_RL.multi_stage_sadpg_wrapper import (
    MultiStageSADPGWrapper,
    DummyMultiStageSADPG,
    create_multi_stage_wrapper,
    FoldingStage,
)


class HierarchicalILFoldEnv(gym.Env):
    """
    Hierarchical Gymnasium environment with IL-based primitives.
    
    The agent selects HIGH-LEVEL stages (stage_1, stage_2, stage_3)
    and the environment executes them using trained SADP_G models.
    
    Action Space (Discrete - 7 actions):
        0: STAGE_1_LEFT_SLEEVE  - Execute SADP_G stage 1 model
        1: STAGE_2_RIGHT_SLEEVE - Execute SADP_G stage 2 model
        2: STAGE_3_BOTTOM_FOLD  - Execute SADP_G stage 3 model
        3: OPEN_HANDS           - Open both grippers
        4: MOVE_TO_HOME         - Move arms to home position
        5: WAIT                 - Wait (do nothing) - allows RL to learn timing
        6: DONE                 - Signal task completion
        
    Observation Space (Dict - ENHANCED):
        - garment_pcd: Point cloud of garment (N, 3)
        - gam_keypoints: GAM-detected manipulation points (6, 3)
        - primitive_mask: Which primitives are still available (7,)
        - executed_sequence: One-hot of executed primitives (7,)
        - overall_fold_quality: Current fold quality [0, 1] (NEW!)
        - stages_completed: Which stages are done [3,] (NEW!)
        - stage_progress: Overall progress [0, 1] (NEW!)
        
    Reward (ENHANCED - Quality-Based):
        - +10.0 base for each successful IL stage
        - +5.0 × fold_quality bonus (quality-based reward)
        - +1.0, +2.0, +3.0 progressive bonus for stages 1, 2, 3
        - +10.0 bonus for completing all three stages
        - +20.0 base + 10.0 × final_quality for task completion
        - -2.0 for failed stage
        - -0.5 for selecting already-executed stage
        - -5.0 for selecting DONE before task is complete
        - -0.1 for WAIT (small penalty to encourage action)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        # SADP_G configuration
        training_data_num: int = 100,
        stage_1_checkpoint: int = 1500,
        stage_2_checkpoint: int = 1500,
        stage_3_checkpoint: int = 1500,
        use_dummy_il: bool = False,
        # Environment configuration
        config: Optional[Dict] = None,
        render_mode: str = "human",
        max_primitives: int = 10,
        point_cloud_size: int = 2048,
        # Logging configuration
        verbose: bool = True,          # Print training progress
        # Device
        device: str = "cuda:0",
    ):
        """
        Initialize hierarchical IL environment.
        
        Args:
            training_data_num: Training data config for SADP_G checkpoints
            stage_1_checkpoint: Checkpoint for stage 1 SADP_G
            stage_2_checkpoint: Checkpoint for stage 2 SADP_G
            stage_3_checkpoint: Checkpoint for stage 3 SADP_G
            use_dummy_il: Use dummy IL policy for testing
            config: Configuration dict (usd_path, ground_material_usd)
            render_mode: "human" or "rgb_array"
            max_primitives: Maximum number of primitive selections per episode
            point_cloud_size: Number of points in observation point cloud
            device: Device for IL policy inference
        """
        super().__init__()
        
        # Store SADP_G configuration
        self.training_data_num = training_data_num
        self.stage_checkpoints = [stage_1_checkpoint, stage_2_checkpoint, stage_3_checkpoint]
        self.use_dummy_il = use_dummy_il
        self.device = device
        
        self.config = config or {}
        self.render_mode = render_mode
        self.max_primitives = max_primitives
        self.point_cloud_size = point_cloud_size
        self.verbose = verbose
        
        # Will be initialized lazily
        self._initialized = False
        self._primitives = None
        self._il_wrapper = None
        
        # Episode state
        self.current_step = 0
        self.total_physics_steps = 0
        self.executed_sequence = []
        self._episode_count = 0
        self._stage_qualities = []  # Track quality after each stage
        
        # Define action space: 7 discrete primitive choices (added WAIT)
        self.action_space = spaces.Discrete(len(ILPrimitiveID))
        
        # Define observation space (ENHANCED with quality and progress metrics)
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
                shape=(len(ILPrimitiveID),),
                dtype=np.float32
            ),
            "executed_sequence": spaces.Box(
                low=0, high=1,
                shape=(len(ILPrimitiveID),),
                dtype=np.float32
            ),
            # NEW: Quality and progress metrics to help RL learn
            "overall_fold_quality": spaces.Box(
                low=0.0, high=1.0,
                shape=(1,),
                dtype=np.float32
            ),
            "stages_completed": spaces.Box(
                low=0, high=1,
                shape=(3,),  # [stage1_done, stage2_done, stage3_done]
                dtype=np.float32
            ),
            "stage_progress": spaces.Box(
                low=0.0, high=1.0,
                shape=(1,),  # Overall progress toward completion
                dtype=np.float32
            ),
        })
        
        # Track initial garment state for reward computation
        self._initial_bbox = None
        
    def _lazy_init(self):
        """Initialize Isaac Sim environment and IL policies lazily."""
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
        from Env_Config.Utils_Project.Code_Tools import normalize_columns
        
        self._set_prim_visible_group = set_prim_visible_group
        self._normalize_columns = normalize_columns
        
        print("[HierarchicalILFoldEnv] Initializing environment...")
        
        # Create base environment
        self._base_env = BaseEnv()
        
        # Add ground
        self._ground = Real_Ground(
            self._base_env.scene,
            visual_material_usd=self.config.get("ground_material_usd"),
        )
        
        # Add garment - MUST use Tops garment to match SADP_G training!
        # Default Particle_Garment uses Dress, but we need Tops for Fold_Tops task
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
        print(f"[HierarchicalILFoldEnv] Using garment: {garment_usd}")
        
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
        
        # Load multi-stage IL wrapper (SADP_G models)
        print(f"[HierarchicalILFoldEnv] Loading SADP_G wrapper (dummy={self.use_dummy_il})...")
        self._il_wrapper = create_multi_stage_wrapper(
            use_dummy=self.use_dummy_il,
            training_data_num=self.training_data_num,
            stage_1_checkpoint=self.stage_checkpoints[0],
            stage_2_checkpoint=self.stage_checkpoints[1],
            stage_3_checkpoint=self.stage_checkpoints[2],
            device=self.device,
            lazy_load=False,
        )
        
        # Create IL-based manipulation primitives
        self._primitives = ILManipulationPrimitives(
            robot=self._robot,
            garment=self._garment,
            base_env=self._base_env,
            garment_camera=self._garment_camera,
            env_camera=self._env_camera,
            gam_model=self._gam_model,
            il_wrapper=self._il_wrapper,
            set_prim_visible_group_func=self._set_prim_visible_group,
            normalize_columns_func=self._normalize_columns,
        )
        
        self._initialized = True
        print("[HierarchicalILFoldEnv] Environment initialized with SADP_G models!")
        
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
        self._stage_qualities = []
        self._episode_count += 1
        self._primitives.reset()
        
        if self.verbose:
            print(f"\n🎬 Starting Episode {self._episode_count}")
            print(f"   IL Policy: {'Dummy' if self.use_dummy_il else 'SADP_G'}")
        
        # ========== CRITICAL: Full World Reset ==========
        # Particle cloth maintains internal state that carries over between episodes.
        # Must call world.reset() to fully reset the particle system.
        self._base_env.reset()
        
        # Fixed garment position (matches SADP_G validation)
        if seed is not None:
            np.random.seed(seed)
        
        # Use fixed position to match SADP_G training distribution
        pos_x = 0.0  # Center
        pos_y = 0.8  # Standard workspace position
        pos = np.array([pos_x, pos_y, 0.2])
        ori = np.array([0.0, 0.0, 0.0])
        
        # Reset garment pose AFTER world reset
        self._garment.set_pose(pos=pos, ori=ori)
        
        # Reset robot to initial configuration
        self._robot.dexleft.post_reset()
        self._robot.dexright.post_reset()
        self._robot.set_both_hand_state("open", "open")
        
        # Step to settle - let garment drop and stabilize
        for _ in range(100):
            self._base_env.step()
        
        # Get initial observation
        obs = self._get_observation()
        
        # Store initial bbox for reward computation
        self._initial_bbox = self._compute_bbox(obs["garment_pcd"])
        
        info = {
            "initial_pos": pos,
            "garment_usd": self._garment.usd_path,
            "il_policy": "SADP_G" if not self.use_dummy_il else "Dummy",
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute a high-level IL primitive selection.
        
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
        primitive_id = ILPrimitiveID(action)
        
        # Get current garment point cloud
        garment_pcd = self._get_garment_pcd()
        
        # Check if primitive was already executed (except DONE and utility primitives)
        already_executed = primitive_id in self._primitives.get_executed_primitives()
        is_folding_primitive = primitive_id in {
            ILPrimitiveID.STAGE_1_LEFT_SLEEVE,
            ILPrimitiveID.STAGE_2_RIGHT_SLEEVE,
            ILPrimitiveID.STAGE_3_BOTTOM_FOLD
        }
        
        # Initialize reward and info
        reward = 0.0
        info = {
            "step": self.current_step,
            "primitive": primitive_id.name,
            "primitive_id": int(primitive_id),
        }
        
        # Handle WAIT action (NEW!)
        if primitive_id == ILPrimitiveID.WAIT:
            # Small negative reward for waiting (encourages action, but allows timing)
            reward = -0.1
            info["message"] = "Waiting..."
            obs = self._get_observation()
            return obs, reward, False, False, info
        
        # Handle DONE action
        if primitive_id == ILPrimitiveID.DONE:
            if self._primitives.is_complete_sequence():
                # Task completed successfully!
                obs = self._get_observation()
                final_quality = self._compute_fold_quality(obs["garment_pcd"])
                
                # Enhanced reward: base + quality bonus
                reward = 20.0  # Base completion bonus (increased from 15.0)
                reward += final_quality * 10.0  # Quality bonus (up to +10)
                
                info["success"] = True
                info["final_quality"] = final_quality
                info["message"] = f"Task completed! Final quality: {final_quality:.3f}"
                
                if self.verbose:
                    self._print_episode_summary(info, True)
                
                return obs, reward, True, False, info
            else:
                # Premature DONE - penalty
                reward = -5.0  # Increased penalty
                info["success"] = False
                info["message"] = "DONE selected but task not complete"
                obs = self._get_observation()
                return obs, reward, False, False, info
        
        # Penalty for re-selecting already executed folding primitive
        if already_executed and is_folding_primitive:
            reward = -0.5
            info["message"] = f"Stage {primitive_id.name} already executed"
            obs = self._get_observation()
            return obs, reward, False, False, info
        
        # Execute the IL primitive (runs SADP_G model)
        result = self._primitives.execute(
            primitive_id=primitive_id,
            garment_pcd=garment_pcd,
        )
        
        self.total_physics_steps += result.steps_taken
        info.update(result.info)
        info["physics_steps"] = result.steps_taken
        
        # Compute reward based on primitive result (ENHANCED REWARD SHAPING)
        if result.success:
            if is_folding_primitive:
                obs = self._get_observation()
                fold_quality = self._compute_fold_quality(obs["garment_pcd"])
                self._stage_qualities.append(fold_quality)
                
                # Enhanced quality-based reward
                base_reward = 10.0  # Base reward for successful stage (increased from 5.0)
                quality_bonus = fold_quality * 5.0  # Quality bonus (increased from 3.0)
                reward = base_reward + quality_bonus
                
                # Progressive bonus: later stages worth more (encourages completion)
                stage_num = int(primitive_id) + 1
                stage_bonus = stage_num * 1.0  # +1, +2, +3 for stages 1, 2, 3
                reward += stage_bonus
                
                info["fold_quality"] = fold_quality
                info["stage_bonus"] = stage_bonus
                
                if self.verbose:
                    print(f"  ✅ Stage {stage_num} completed! Quality: {fold_quality:.3f}, Reward: {reward:.2f}")
            else:
                reward = 0.2  # Small reward for utility primitives (increased from 0.1)
        else:
            reward = -2.0  # Penalty for failed primitive (increased from -1.0)
            info["error"] = result.info.get("error", "Unknown error")
            if self.verbose:
                print(f"  ❌ Stage failed: {info.get('error', 'Unknown error')}")
        
        # Check if all stages done (but agent hasn't said DONE yet)
        if self._primitives.is_complete_sequence():
            reward += 10.0  # Bonus for completing all IL stages (increased from 8.0)
            info["all_stages_complete"] = True
            if self.verbose:
                avg_quality = np.mean(self._stage_qualities) if self._stage_qualities else 0.0
                print(f"  🎉 All stages complete! Average quality: {avg_quality:.3f}")
        
        # Track sequence
        self.executed_sequence.append(primitive_id)
        
        # Get new observation
        obs = self._get_observation()
        
        # Check termination conditions
        terminated = False  # Only terminate when agent says DONE
        truncated = self.current_step >= self.max_primitives
        
        if truncated:
            info["message"] = "Episode truncated (max primitives reached)"
            if self.verbose:
                self._print_episode_summary(info, False, True)
        
        return obs, reward, terminated, truncated, info
    
    def _print_episode_summary(self, info: Dict, success: bool, truncated: bool = False):
        """Print episode summary at the end."""
        stages_done = sum([
            ILPrimitiveID.STAGE_1_LEFT_SLEEVE in self._primitives.get_executed_primitives(),
            ILPrimitiveID.STAGE_2_RIGHT_SLEEVE in self._primitives.get_executed_primitives(),
            ILPrimitiveID.STAGE_3_BOTTOM_FOLD in self._primitives.get_executed_primitives(),
        ])
        
        avg_quality = np.mean(self._stage_qualities) if self._stage_qualities else 0.0
        final_quality = info.get("final_quality", self._compute_fold_quality(self._get_garment_pcd()))
        
        outcome = "✅ SUCCESS" if success else ("⏱️ TRUNCATED" if truncated else "❌ FAILED")
        
        print(f"\n{'='*60}")
        print(f"📊 Episode {self._episode_count} Summary")
        print(f"{'='*60}")
        print(f"  Outcome:        {outcome}")
        print(f"  Steps:          {self.current_step}")
        print(f"  Stages Done:    {stages_done}/3")
        print(f"  Avg Quality:    {avg_quality:.3f}")
        print(f"  Final Quality:  {final_quality:.3f}")
        print(f"  Sequence:       {' → '.join([p.name for p in self.executed_sequence])}")
        print(f"{'='*60}\n")
    
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
        primitive_mask = np.ones(len(ILPrimitiveID), dtype=np.float32)
        for pid in executed:
            if pid in {ILPrimitiveID.STAGE_1_LEFT_SLEEVE, 
                      ILPrimitiveID.STAGE_2_RIGHT_SLEEVE, 
                      ILPrimitiveID.STAGE_3_BOTTOM_FOLD}:
                primitive_mask[pid] = 0.0  # IL stages can only be done once
        
        # Build executed sequence (one-hot)
        executed_sequence = np.zeros(len(ILPrimitiveID), dtype=np.float32)
        for pid in executed:
            executed_sequence[pid] = 1.0
        
        # Compute quality and progress metrics (NEW!)
        overall_quality = self._compute_fold_quality(garment_pcd)
        
        # Stages completed (one-hot for each of the 3 folding stages)
        stages_completed = np.zeros(3, dtype=np.float32)
        if ILPrimitiveID.STAGE_1_LEFT_SLEEVE in executed:
            stages_completed[0] = 1.0
        if ILPrimitiveID.STAGE_2_RIGHT_SLEEVE in executed:
            stages_completed[1] = 1.0
        if ILPrimitiveID.STAGE_3_BOTTOM_FOLD in executed:
            stages_completed[2] = 1.0
        
        # Overall progress (0.0 to 1.0)
        stage_progress = np.sum(stages_completed) / 3.0
        
        obs = {
            "garment_pcd": garment_pcd.astype(np.float32),
            "gam_keypoints": gam_keypoints,
            "primitive_mask": primitive_mask,
            "executed_sequence": executed_sequence,
            # NEW: Quality and progress metrics
            "overall_fold_quality": np.array([overall_quality], dtype=np.float32),
            "stages_completed": stages_completed,
            "stage_progress": np.array([stage_progress], dtype=np.float32),
        }
        
        # NaN protection
        for key, value in obs.items():
            if np.isnan(value).any() or np.isinf(value).any():
                obs[key] = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        return obs
    
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
            print("[HierarchicalILFoldEnv] Environment closed.")


def register_hierarchical_il_env():
    """Register environment with Gymnasium."""
    gym.register(
        id="HierarchicalILFoldTops-v0",
        entry_point="Env_RL.hierarchical_il_fold_env:HierarchicalILFoldEnv",
        max_episode_steps=10,
    )
