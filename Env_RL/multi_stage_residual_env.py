"""
Multi-Stage Residual RL Environment for Garment Folding with SADP_G Guidance.

This environment implements multi-stage residual RL where:
- 3 SADP_G models (frozen) provide stage-specific IL guidance
- Single RL policy learns residuals AND when to transition stages
- Stage transitions are learned, not hard-coded

Architecture:
    action = [joint_residuals(60D), stage_advance_signal(1D)]
    final_joint_action = IL_joint_action(current_stage) + surgical_residuals
    
    NOTE: Actions are in JOINT SPACE (60D) to match SADP_G outputs exactly.
    This ensures residuals are meaningful corrections to the IL policy.

Key Features:
- Single RL policy handles all 3 stages
- Stage indicator in observation enables stage-aware behavior
- RL learns WHEN to advance stages (not just how to fold)
- Bounded residuals ensure IL provides strong guidance
- Stage-specific rewards encourage proper sequencing
- Joint-space actions match SADP_G output format

=== Quick Wins: Surgical Residual Application ===
Since IL is already performing well, RL makes SURGICAL interventions:

1. ARM-ONLY RESIDUALS (arm_only_residuals=True):
   - Only arm joints (6 per side = 12 total) receive residuals
   - Hand joints (24 per side) are 100% IL-controlled
   - This preserves IL's grasp configuration

2. SPARSE APPLICATION (residual_apply_interval=5):
   - Residuals only applied every N steps
   - Reduces noise and lets IL trajectory shine through
   - RL focuses on key positioning moments

3. SKIP DURING MANIPULATION (disable_residual_during_manipulation=True):
   - When gripper is closed (grasping), use pure IL
   - Residuals only affect approach/positioning phases
   - Protects delicate manipulation

Usage:
    env = MultiStageResidualEnv(
        training_data_num=100,
        stage_checkpoints=[1500, 1500, 1500],
    )
    obs, info = env.reset()
    
    # Action: [joint_residuals(60D), advance_stage(1D)]
    action = rl_policy(obs)
    next_obs, reward, done, truncated, info = env.step(action)
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any, List, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Env_RL.multi_stage_sadpg_wrapper import (
    MultiStageSADPGWrapper,
    DummyMultiStageSADPG,
    create_multi_stage_wrapper,
    FoldingStage,
    STAGE_CONFIGS,
)


class MultiStageResidualEnv(gym.Env):
    """
    Multi-Stage Residual RL Environment with learned stage transitions.
    
    The RL agent outputs:
    - Joint-space residual corrections to add to IL actions (60D)
    - Stage advance signal (1D) - whether to transition to next stage
    
    The environment:
    - Tracks current folding stage (1, 2, or 3)
    - Uses appropriate SADP_G model for IL guidance
    - Rewards good folding AND good timing of stage transitions
    
    IMPORTANT: Actions are in JOINT SPACE (60D) to match SADP_G outputs.
    This ensures residuals are meaningful corrections.
    
    Action Space (61D continuous):
        [0:30]  left arm+hand joint residuals (6 arm + 24 hand DOF)
        [30:60] right arm+hand joint residuals (6 arm + 24 hand DOF)
        [60]    stage_advance_signal (>0.5 = attempt advance)
        
    Observation Space (Dict):
        - garment_pcd: (2048, 3) garment point cloud
        - joint_positions: (60,) current joint positions
        - ee_poses: (14,) end-effector poses
        - il_action: (60,) proposed JOINT action from current stage's SADP_G
        - gam_keypoints: (6, 3) manipulation keypoints
        - current_stage: (4,) one-hot encoding [stage1, stage2, stage3, completed]
        - stage_progress: (1,) normalized progress in current stage
        - stages_completed: (3,) binary mask of completed stages
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
        max_episode_steps: int = 400,  # Total across all stages
        # Residual configuration - now with separate arm/hand scales
        arm_residual_scale: float = 0.05,   # For 6 arm joints per side
        hand_residual_scale: float = 0.01,  # For 24 hand joints per side (more sensitive!)
        action_scale: float = 0.05,
        # ========== Quick Wins: Surgical Residual Application ==========
        arm_only_residuals: bool = True,          # Only apply residuals to arm (not hand)
        residual_apply_interval: int = 5,         # Apply residual every N steps
        disable_residual_during_manipulation: bool = True,  # Skip residuals when grasping
        gripper_threshold: float = 0.5,           # Threshold for gripper closed detection
        # ========== Initial State Configuration ==========
        # Match SADP_G training distribution to avoid data shift!
        use_fixed_initial_state: bool = True,     # Use fixed position (matches validation)
        fixed_pos_x: float = 0.0,                 # Fixed X position (center)
        fixed_pos_y: float = 0.8,                 # Fixed Y position
        random_pos_x_range: Tuple[float, float] = (-0.1, 0.1),  # Random X range if not fixed
        random_pos_y_range: Tuple[float, float] = (0.7, 0.9),   # Random Y range if not fixed
        # Observation configuration
        point_cloud_size: int = 2048,
        # Stage transition configuration
        stage_advance_threshold: float = 0.5,
        min_steps_before_advance: int = 10,  # Min steps in stage before can advance
        # Device
        device: str = "cuda:0",
    ):
        """
        Initialize multi-stage residual environment.
        
        Args:
            training_data_num: Training data config for SADP_G checkpoints
            stage_1_checkpoint: Checkpoint for stage 1 SADP_G
            stage_2_checkpoint: Checkpoint for stage 2 SADP_G
            stage_3_checkpoint: Checkpoint for stage 3 SADP_G
            use_dummy_il: Use dummy IL policy for testing
            config: Additional environment config
            render_mode: Rendering mode
            max_episode_steps: Maximum total steps
            arm_residual_scale: Maximum residual for arm joints (larger ok)
            hand_residual_scale: Maximum residual for hand joints (smaller, more sensitive)
            action_scale: Scale for converting to robot commands
            arm_only_residuals: If True, only arm joints get residuals (hand = pure IL)
            residual_apply_interval: Apply residuals every N steps (sparse intervention)
            disable_residual_during_manipulation: If True, skip residuals when gripper closed
            gripper_threshold: Threshold for detecting gripper closed state
            use_fixed_initial_state: If True, use fixed position (matches SADP_G validation)
            fixed_pos_x: Fixed X position for garment (default 0.0 = center)
            fixed_pos_y: Fixed Y position for garment (default 0.8)
            random_pos_x_range: (min, max) X range if using random position
            random_pos_y_range: (min, max) Y range if using random position
            point_cloud_size: Number of points in observation
            stage_advance_threshold: Threshold for stage advance signal
            min_steps_before_advance: Minimum steps before stage can advance
            device: Device for IL policy inference
        """
        super().__init__()
        
        # Store configuration
        self.training_data_num = training_data_num
        self.stage_checkpoints = [stage_1_checkpoint, stage_2_checkpoint, stage_3_checkpoint]
        self.use_dummy_il = use_dummy_il
        self.config = config or {}
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.arm_residual_scale = arm_residual_scale
        self.hand_residual_scale = hand_residual_scale
        self.action_scale = action_scale
        self.point_cloud_size = point_cloud_size
        self.stage_advance_threshold = stage_advance_threshold
        self.min_steps_before_advance = min_steps_before_advance
        self.device = device
        
        # Will be initialized lazily
        self._initialized = False
        self._il_wrapper = None
        
        # Episode state
        self.current_step = 0
        self._initial_bbox = None
        self._stage_initial_bbox = None  # BBox at start of current stage
        self._last_il_action = None
        self._last_joint_action = None  # For smoothness penalty
        self._stages_completed = [False, False, False]
        
        # Progress tracking for early termination
        self._recent_fold_progress = []
        self._progress_window = 30  # Track last 30 steps
        
        # ========== Quick Wins: Surgical Residual Application ==========
        # 1. Only apply residuals to ARM joints (not hand) - hand is IL-controlled
        self.arm_only_residuals = arm_only_residuals
        
        # 2. Only apply residuals every N steps (sparse intervention)
        self.residual_apply_interval = residual_apply_interval
        
        # 3. Disable residuals during active manipulation (gripper closed)
        self.disable_residual_during_manipulation = disable_residual_during_manipulation
        self._gripper_threshold = gripper_threshold
        
        # ========== Initial State Configuration ==========
        # Fix distribution shift: match SADP_G training distribution
        self.use_fixed_initial_state = use_fixed_initial_state
        self.fixed_pos_x = fixed_pos_x
        self.fixed_pos_y = fixed_pos_y
        self.random_pos_x_range = random_pos_x_range
        self.random_pos_y_range = random_pos_y_range
        
        # Joint space dimensions: 30 per arm+hand (6 arm + 24 hand)
        self.joint_dim = 60
        self.arm_dof = 6   # UR10e arm joints
        self.hand_dof = 24 # Shadow hand joints
        
        # Build non-uniform residual bounds (arm vs hand)
        # Structure: [left_arm(6), left_hand(24), right_arm(6), right_hand(24), stage_advance(1)]
        low_bounds = np.concatenate([
            [-arm_residual_scale] * self.arm_dof,    # Left arm
            [-hand_residual_scale] * self.hand_dof,  # Left hand (smaller!)
            [-arm_residual_scale] * self.arm_dof,    # Right arm
            [-hand_residual_scale] * self.hand_dof,  # Right hand (smaller!)
            [-1.0]  # Stage advance signal
        ])
        high_bounds = np.concatenate([
            [arm_residual_scale] * self.arm_dof,
            [hand_residual_scale] * self.hand_dof,
            [arm_residual_scale] * self.arm_dof,
            [hand_residual_scale] * self.hand_dof,
            [1.0]
        ])
        
        # Action space: joint_residuals(60D) + stage_advance(1D)
        # Different bounds for arm vs hand joints!
        self.action_space = spaces.Box(
            low=low_bounds.astype(np.float32),
            high=high_bounds.astype(np.float32),
            dtype=np.float32
        )
        
        # Store bounds for clipping
        self._residual_low = low_bounds[:self.joint_dim]
        self._residual_high = high_bounds[:self.joint_dim]
        
        # Observation space
        self.observation_space = spaces.Dict({
            "garment_pcd": spaces.Box(
                low=-10.0, high=10.0,
                shape=(point_cloud_size, 3),
                dtype=np.float32
            ),
            "joint_positions": spaces.Box(
                low=-2 * np.pi, high=2 * np.pi,
                shape=(self.joint_dim,),
                dtype=np.float32
            ),
            "ee_poses": spaces.Box(
                low=-10.0, high=10.0,
                shape=(14,),
                dtype=np.float32
            ),
            # IL action is now 60D joint space (same as SADP_G output)
            "il_action": spaces.Box(
                low=-2 * np.pi, high=2 * np.pi,
                shape=(self.joint_dim,),
                dtype=np.float32
            ),
            "gam_keypoints": spaces.Box(
                low=-10.0, high=10.0,
                shape=(6, 3),
                dtype=np.float32
            ),
            "current_stage": spaces.Box(
                low=0, high=1,
                shape=(4,),  # [stage1, stage2, stage3, completed]
                dtype=np.float32
            ),
            "stage_progress": spaces.Box(
                low=0, high=1,
                shape=(1,),
                dtype=np.float32
            ),
            "stages_completed": spaces.Box(
                low=0, high=1,
                shape=(3,),  # [stage1_done, stage2_done, stage3_done]
                dtype=np.float32
            ),
        })
        
        # Reward weights
        self.reward_weights = {
            # Task rewards
            "fold_progress": 1.0,
            "compactness": 0.5,
            "height_penalty": 0.3,
            # Residual rewards
            "residual_penalty": 0.02,
            "smoothness_penalty": 0.05,  # Penalize jerky movements
            # Stage transition rewards
            "stage_advance_bonus": 5.0,
            "premature_advance_penalty": 2.0,
            "late_advance_penalty": 0.1,
            # Completion rewards
            "stage_completion_bonus": 3.0,
            "task_success_bonus": 20.0,
            # Early termination penalty
            "early_termination_penalty": 5.0,
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
        from Env_Config.Utils_Project.Code_Tools import normalize_columns
        
        self._set_prim_visible_group = set_prim_visible_group
        self._normalize_columns = normalize_columns
        
        print("[MultiStageResidualEnv] Initializing environment...")
        
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
        print(f"[MultiStageResidualEnv] Using garment: {garment_usd}")
        
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
        
        # Load multi-stage IL wrapper
        print(f"[MultiStageResidualEnv] Loading SADP_G wrapper (dummy={self.use_dummy_il})...")
        self._il_wrapper = create_multi_stage_wrapper(
            use_dummy=self.use_dummy_il,
            training_data_num=self.training_data_num,
            stage_1_checkpoint=self.stage_checkpoints[0],
            stage_2_checkpoint=self.stage_checkpoints[1],
            stage_3_checkpoint=self.stage_checkpoints[2],
            device=self.device,
            lazy_load=False,  # Load immediately
        )
        
        # Cache for current affordance features
        self._current_affordance = None
        self._current_garment_pcd = None
        self._gam_keypoints = None
        self._gam_similarity = None
        
        # ========== Store Initial Garment State ==========
        # Store initial particle positions so we can reset to extended shape
        # This is critical because world.reset() may not fully reset particle cloth!
        self._store_initial_garment_state()
        
        self._initialized = True
        print("[MultiStageResidualEnv] Environment initialized!")
    
    def _store_initial_garment_state(self):
        """
        Store the initial garment particle positions (extended shape).
        
        This is called once during initialization to capture the garment's
        rest shape before any simulation. We restore this on each reset
        because world.reset() may not fully reset particle cloth state.
        """
        try:
            # Get particle positions from cloth prim
            # The garment_mesh has a _cloth_prim_view that gives access to particles
            positions = self._garment.get_vertice_positions()
            if positions is not None:
                self._initial_particle_positions = positions.copy()
                print(f"[MultiStageResidualEnv] Stored initial garment shape: {len(positions)} particles")
            else:
                self._initial_particle_positions = None
                print("[MultiStageResidualEnv] Warning: Could not get initial particle positions")
        except Exception as e:
            print(f"[MultiStageResidualEnv] Warning: Failed to store initial garment state: {e}")
            self._initial_particle_positions = None
    
    def _reset_garment_to_initial_shape(self):
        """
        Reset garment particles to their initial extended shape.
        
        This ensures the cloth starts in the same configuration every episode,
        regardless of how it ended up in the previous episode.
        
        Note: This is a best-effort reset. Isaac Sim particle cloth can be
        tricky to reset fully. If this doesn't work perfectly, increase
        settle time or use stronger gravity during settling.
        """
        if self._initial_particle_positions is None:
            return
        
        try:
            # Try to reset particle positions through the cloth prim view
            # This is Isaac Sim specific and may need adjustment based on version
            cloth_view = self._garment.particle_controller
            if cloth_view is not None and hasattr(cloth_view, 'set_world_positions'):
                cloth_view.set_world_positions(self._initial_particle_positions)
            elif hasattr(self._garment.garment_mesh, 'set_points_positions'):
                self._garment.garment_mesh.set_points_positions(self._initial_particle_positions)
        except Exception as e:
            # If explicit reset fails, we rely on world.reset() + settle time
            pass
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self._lazy_init()
        
        # Reset episode state
        self.current_step = 0
        self._stages_completed = [False, False, False]
        self._il_wrapper.reset()
        
        # Reset tracking variables
        self._last_joint_action = None
        self._recent_fold_progress = []
        
        # ========== CRITICAL: Full World Reset ==========
        # The particle cloth maintains internal state (velocities, rotations) that
        # carries over between episodes. We MUST call world.reset() to fully reset
        # the particle system, then re-set the garment pose.
        self._base_env.reset()  # Resets world including particle system!
        
        # ========== Reset Garment to Extended Shape ==========
        # Try to reset particle positions to initial (extended) shape
        # This is important because world.reset() may not fully reset cloth state
        self._reset_garment_to_initial_shape()
        
        # ========== Initial State: Fixed or Random ==========
        # Use fixed position to match SADP_G training distribution (avoids data shift!)
        if seed is not None:
            np.random.seed(seed)
        
        if self.use_fixed_initial_state:
            # Fixed position - matches SADP_G validation setup exactly
            pos_x = self.fixed_pos_x
            pos_y = self.fixed_pos_y
        else:
            # Random position - only use if SADP_G was trained with randomization
            pos_x = np.random.uniform(self.random_pos_x_range[0], self.random_pos_x_range[1])
            pos_y = np.random.uniform(self.random_pos_y_range[0], self.random_pos_y_range[1])
        
        pos = np.array([pos_x, pos_y, 0.2])
        ori = np.array([0.0, 0.0, 0.0])
        
        # Set garment pose AFTER world reset and shape reset
        self._garment.set_pose(pos=pos, ori=ori)
        
        # Reset robot to initial configuration
        self._robot.dexleft.post_reset()
        self._robot.dexright.post_reset()
        self._robot.set_both_hand_state("open", "open")
        
        # ========== Settle with High Gravity ==========
        # Use high gravity to help cloth settle quickly into extended shape
        self._garment.particle_material.set_gravity_scale(10.0)
        for _ in range(50):
            self._base_env.step()
        
        # Return to normal gravity and settle more
        self._garment.particle_material.set_gravity_scale(1.0)
        for _ in range(100):
            self._base_env.step()
        
        # Get initial garment state
        self._update_garment_pcd_and_gam()
        
        # Store initial state
        self._initial_bbox = self._compute_bbox(self._current_garment_pcd)
        self._stage_initial_bbox = self._initial_bbox.copy()
        
        # ========== Pre-position Robot to First Manipulation Point ==========
        # CRITICAL: The original SADP_G validation moves the robot to the
        # manipulation point BEFORE starting inference. Without this step,
        # the robot starts too far from the garment.
        self._pre_position_robot_for_stage()
        
        # Get observation
        obs = self._get_observation()
        
        info = {
            "initial_pos": pos,
            "initial_state_mode": "fixed" if self.use_fixed_initial_state else "random",
            "stage": self._il_wrapper.get_stage_info(),
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute step with joint-space residual action and stage advance signal.
        
        Args:
            action: [joint_residuals(60D), stage_advance(1D)]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        # Parse action: 60D joint residuals + 1D stage advance
        # Use non-uniform clipping (different bounds for arm vs hand)
        residual_action = np.clip(action[:self.joint_dim], self._residual_low, self._residual_high)
        stage_advance_signal = action[self.joint_dim]
        
        # Get IL policy's joint-space action for current stage
        il_obs = self._get_il_observation()
        il_action = self._get_il_joint_action(il_obs)  # 60D joint positions
        self._last_il_action = il_action
        
        # ========== Quick Wins: Surgical Residual Application ==========
        # Decide whether to apply residuals this step
        apply_residual = self._should_apply_residual()
        
        if apply_residual:
            # Apply surgical residual (arm-only if enabled)
            effective_residual = self._get_surgical_residual(residual_action)
        else:
            # Pure IL - no residual
            effective_residual = np.zeros_like(residual_action)
        
        # Combine in joint space: final = IL + residual
        final_joint_action = il_action + effective_residual
        
        # Execute robot action using joint positions directly
        self._execute_joint_action(final_joint_action)
        
        # Store for smoothness penalty
        self._last_joint_action = final_joint_action.copy()
        
        # Update IL observation history (important for temporal diffusion policy)
        self._il_wrapper.update_obs(il_obs)
        
        # Handle stage advance
        stage_advanced = False
        advance_reward = 0.0
        
        if stage_advance_signal > self.stage_advance_threshold:
            stage_advanced, advance_reward = self._handle_stage_advance()
        
        # Update garment state
        self._update_garment_pcd_and_gam()
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward (includes smoothness penalty)
        # Use effective_residual for reward so we only penalize residuals that were actually applied
        reward, reward_info = self._compute_reward(obs, effective_residual, stage_advanced, final_joint_action)
        reward += advance_reward
        reward_info["advance_reward"] = advance_reward
        reward_info["residual_applied"] = apply_residual
        reward_info["effective_residual_norm"] = np.linalg.norm(effective_residual)
        
        # Track progress for early termination
        self._recent_fold_progress.append(reward_info.get("fold_progress", 0))
        if len(self._recent_fold_progress) > self._progress_window:
            self._recent_fold_progress.pop(0)
        
        # Check for early termination (bad exploration)
        early_term, term_reason = self._check_early_termination(obs)
        
        # Check termination
        all_stages_done = all(self._stages_completed)
        terminated = all_stages_done and self._check_final_success(obs)
        
        # Early termination overrides normal termination
        if early_term:
            terminated = True
            reward -= self.reward_weights["early_termination_penalty"]
            reward_info["early_termination"] = True
            reward_info["termination_reason"] = term_reason
        
        truncated = self.current_step >= self.max_episode_steps
        
        info = {
            "step": self.current_step,
            "stage_info": self._il_wrapper.get_stage_info(),
            "stages_completed": self._stages_completed.copy(),
            "stage_advanced": stage_advanced,
            "il_action_norm": np.linalg.norm(il_action),
            "residual_norm": np.linalg.norm(residual_action),
            "is_success": terminated and all_stages_done and not early_term,
            **reward_info,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _execute_joint_action(self, joint_action: np.ndarray):
        """
        Execute joint-space action on robot.
        
        This matches the execution pattern in Fold_Tops_HALO.py where
        SADP_G outputs are applied directly as ArticulationAction.
        
        Args:
            joint_action: (60,) array of target joint positions
                [0:30]  = left arm (6) + left hand (24)
                [30:60] = right arm (6) + right hand (24)
        """
        from isaacsim.core.utils.types import ArticulationAction
        
        # Split into left and right arm+hand
        action_left = ArticulationAction(joint_positions=joint_action[:30])
        action_right = ArticulationAction(joint_positions=joint_action[30:])
        
        # Apply actions to both robots
        self._robot.dexleft.apply_action(action_left)
        self._robot.dexright.apply_action(action_right)
        
        # Step physics multiple times for stability (matches HALO validation)
        for _ in range(5):
            self._base_env.step()
    
    def _should_apply_residual(self) -> bool:
        """
        Determine if residual should be applied this timestep.
        
        Quick Wins logic:
        1. Only apply every N steps (sparse intervention)
        2. Skip during manipulation phase (gripper closed)
        
        Returns:
            True if residual should be applied
        """
        # Check 1: Sparse application - only every N steps
        if self.current_step % self.residual_apply_interval != 0:
            return False
        
        # Check 2: Skip during active manipulation (gripper closed)
        if self.disable_residual_during_manipulation:
            if self._is_gripper_closed():
                return False
        
        return True
    
    def _is_gripper_closed(self) -> bool:
        """
        Check if either gripper is in closed/grasping state.
        
        Uses hand joint positions to determine if fingers are closed.
        Shadow hand: joints 6-29 for left, 36-59 for right
        When grasping, finger joints tend to have larger (more flexed) values.
        
        Returns:
            True if gripper is closed/grasping
        """
        try:
            left_joints = self._robot.dexleft.get_joint_positions()
            right_joints = self._robot.dexright.get_joint_positions()
            
            # Get hand joints (indices 6-29 for each arm+hand)
            left_hand = left_joints[6:30]
            right_hand = right_joints[6:30]
            
            # Simple heuristic: average finger flexion > threshold means grasping
            # Higher values typically = more closed fingers
            left_avg_flexion = np.mean(np.abs(left_hand))
            right_avg_flexion = np.mean(np.abs(right_hand))
            
            return (left_avg_flexion > self._gripper_threshold or 
                    right_avg_flexion > self._gripper_threshold)
        except:
            return False
    
    def _get_surgical_residual(self, residual_action: np.ndarray) -> np.ndarray:
        """
        Get surgically modified residual (arm-only, hand zeroed out).
        
        Quick Win: Only apply residuals to arm joints (6 per arm = 12 total).
        Hand joints are controlled 100% by IL.
        
        Joint structure:
            [0:6]   = left arm
            [6:30]  = left hand (zeroed)
            [30:36] = right arm  
            [36:60] = right hand (zeroed)
        
        Args:
            residual_action: Full 60D residual from RL policy
            
        Returns:
            Surgical residual with hand joints zeroed
        """
        if not self.arm_only_residuals:
            return residual_action
        
        surgical_residual = residual_action.copy()
        
        # Zero out hand joints - let IL control them completely
        surgical_residual[6:30] = 0.0    # Left hand
        surgical_residual[36:60] = 0.0   # Right hand
        
        return surgical_residual
    
    def _handle_stage_advance(self) -> Tuple[bool, float]:
        """
        Handle stage advance attempt.
        
        Returns:
            (success, reward) tuple
        """
        current_stage = self._il_wrapper.current_stage
        stage_steps = self._il_wrapper.stage_step_count
        
        # Check if already completed all stages
        if current_stage == FoldingStage.COMPLETED:
            return False, 0.0
        
        # Check minimum steps requirement
        if stage_steps < self.min_steps_before_advance:
            return False, -self.reward_weights["premature_advance_penalty"]
        
        # Check if current stage fold is good enough
        stage_quality = self._evaluate_current_stage_quality()
        
        if stage_quality < 0.3:
            # Too early to advance
            return False, -self.reward_weights["premature_advance_penalty"]
        
        # Mark current stage as completed
        stage_idx = int(current_stage) - 1
        self._stages_completed[stage_idx] = True
        
        # Advance to next stage
        success = self._il_wrapper.advance_stage()
        
        if success:
            # Store new stage initial bbox
            self._stage_initial_bbox = self._compute_bbox(self._current_garment_pcd)
            
            # Update garment state and GAM for new stage
            self._update_garment_pcd_and_gam()
            
            # Pre-position robot for new stage
            # This is critical - move robot to the new manipulation point!
            self._pre_position_robot_for_stage()
            
            # Calculate reward based on quality
            advance_reward = self.reward_weights["stage_advance_bonus"] * stage_quality
            advance_reward += self.reward_weights["stage_completion_bonus"]
            
            return True, advance_reward
        
        return False, 0.0
    
    def _evaluate_current_stage_quality(self) -> float:
        """
        Evaluate how well the current stage fold has been completed.
        
        Returns:
            Quality score in [0, 1]
        """
        if self._stage_initial_bbox is None:
            return 0.0
        
        current_bbox = self._compute_bbox(self._current_garment_pcd)
        current_size = current_bbox[3:6] - current_bbox[0:3]
        initial_size = self._stage_initial_bbox[3:6] - self._stage_initial_bbox[0:3]
        
        # Calculate area reduction
        initial_area = initial_size[0] * initial_size[1]
        current_area = current_size[0] * current_size[1]
        
        if initial_area < 1e-6:
            return 0.0
        
        # Quality is based on area reduction
        area_reduction = (initial_area - current_area) / initial_area
        
        # Clamp to [0, 1]
        return np.clip(area_reduction, 0.0, 1.0)
    
    def _update_garment_pcd_and_gam(self):
        """Update cached garment point cloud and GAM keypoints."""
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
        
        # Normalize point cloud
        self._current_garment_pcd = self._normalize_pcd(pcd)
        
        # Get GAM keypoints and similarity
        try:
            keypoints, indices, similarity = self._gam_model.get_manipulation_points(
                input_pcd=self._current_garment_pcd,
                index_list=[957, 501, 1902, 448, 1196, 422]
            )
            self._gam_keypoints = keypoints.astype(np.float32)
            self._gam_similarity = similarity
        except:
            self._gam_keypoints = np.zeros((6, 3), dtype=np.float32)
            self._gam_similarity = None
        
        # Update affordance based on current stage
        self._update_affordance_for_stage()
    
    def _pre_position_robot_for_stage(self):
        """
        Pre-position the robot to the manipulation point for the current stage.
        
        CRITICAL: The original SADP_G validation moves the robot to the
        manipulation point BEFORE starting inference using IK:
        
            env.bimanual_dex.dexleft.dense_step_action(target_pos=manipulation_points[0], ...)
            for i in range(20):
                env.step()
        
        Without this step, the robot starts too far from the garment and
        SADP_G actions won't reach the cloth properly.
        """
        if self._gam_keypoints is None:
            return
        
        try:
            current_stage = self._il_wrapper.current_stage
            
            # Get manipulation points for current stage
            # Keypoint mapping:
            # 0: left sleeve tip, 1: left sleeve target
            # 2: right sleeve tip, 3: right sleeve target
            # 4: bottom left, 5: bottom right
            
            if current_stage == FoldingStage.STAGE_1_LEFT_SLEEVE:
                # Move left hand to left sleeve tip
                target_pos = self._gam_keypoints[0].copy()
                target_pos[2] = 0.02  # Slightly above ground
                target_ori = np.array([0.579, -0.579, -0.406, 0.406])
                
                self._robot.dexleft.dense_step_action(
                    target_pos=target_pos,
                    target_ori=target_ori,
                    angular_type="quat"
                )
                
            elif current_stage == FoldingStage.STAGE_2_RIGHT_SLEEVE:
                # Move right hand to right sleeve tip
                target_pos = self._gam_keypoints[2].copy()
                target_pos[2] = 0.02
                target_ori = np.array([0.406, -0.406, -0.579, 0.579])
                
                self._robot.dexright.dense_step_action(
                    target_pos=target_pos,
                    target_ori=target_ori,
                    angular_type="quat"
                )
                
            elif current_stage == FoldingStage.STAGE_3_BOTTOM_UP:
                # Move both hands to bottom corners
                left_pos = self._gam_keypoints[4].copy()
                right_pos = self._gam_keypoints[5].copy()
                left_pos[2] = 0.0
                right_pos[2] = 0.0
                
                self._robot.dense_move_both_ik(
                    left_pos=left_pos,
                    left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
                    right_pos=right_pos,
                    right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
                )
            
            # Settle after pre-positioning
            for _ in range(20):
                self._base_env.step()
                
        except Exception as e:
            print(f"[WARNING] Pre-positioning failed: {e}")
    
    def _update_affordance_for_stage(self):
        """Update affordance features based on current stage."""
        if self._gam_similarity is None:
            self._current_affordance = np.zeros((self.point_cloud_size, 2), dtype=np.float32)
            return
        
        current_stage = self._il_wrapper.current_stage
        config = STAGE_CONFIGS.get(current_stage)
        
        if config is None:
            self._current_affordance = np.zeros((self.point_cloud_size, 2), dtype=np.float32)
            return
        
        # Get affordance indices for this stage
        aff_indices = config.affordance_indices
        
        try:
            # Build affordance features from similarity
            aff = np.stack([
                self._gam_similarity[aff_indices[0]],
                self._gam_similarity[aff_indices[1]]
            ], axis=-1)
            
            # Normalize
            self._current_affordance = self._normalize_columns(aff).astype(np.float32)
        except:
            self._current_affordance = np.zeros((self.point_cloud_size, 2), dtype=np.float32)
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        # Get joint positions
        left_joints = self._robot.dexleft.get_joint_positions()
        right_joints = self._robot.dexright.get_joint_positions()
        joint_positions = np.concatenate([left_joints, right_joints]).astype(np.float32)
        
        # Get EE poses
        left_pos, left_ori = self._robot.dexleft.get_cur_ee_pos()
        right_pos, right_ori = self._robot.dexright.get_cur_ee_pos()
        ee_poses = np.concatenate([left_pos, left_ori, right_pos, right_ori]).astype(np.float32)
        
        # Get IL action (60D joint space - same as SADP_G output)
        if self._last_il_action is not None:
            il_action = self._last_il_action.astype(np.float32)
        else:
            il_obs = self._get_il_observation()
            il_action = self._get_il_joint_action(il_obs).astype(np.float32)
            self._last_il_action = il_action
        
        return {
            "garment_pcd": self._current_garment_pcd.astype(np.float32),
            "joint_positions": joint_positions,
            "ee_poses": ee_poses,
            "il_action": il_action,  # Now 60D joint space
            "gam_keypoints": self._gam_keypoints,
            "current_stage": self._il_wrapper.get_stage_one_hot(),
            "stage_progress": np.array([self._il_wrapper.get_stage_progress()], dtype=np.float32),
            "stages_completed": np.array(self._stages_completed, dtype=np.float32),
        }
    
    def _get_il_observation(self) -> Dict[str, np.ndarray]:
        """Get observation in format expected by SADP_G."""
        left_joints = self._robot.dexleft.get_joint_positions()
        right_joints = self._robot.dexright.get_joint_positions()
        joint_positions = np.concatenate([left_joints, right_joints])
        
        env_pcd = self._env_camera.get_pointcloud_from_depth(
            show_original_pc_online=False,
            show_downsample_pc_online=False,
        )
        
        return {
            "agent_pos": joint_positions,
            "environment_point_cloud": env_pcd,
            "garment_point_cloud": self._current_garment_pcd,
            "points_affordance_feature": self._current_affordance,
        }
    
    def _get_il_joint_action(self, il_obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get IL action directly in joint space (60D).
        
        SADP_G outputs joint-space actions which we use directly.
        This ensures residuals are meaningful corrections to IL.
        
        Args:
            il_obs: Observation dict for IL policy
            
        Returns:
            joint_action: (60,) target joint positions from SADP_G
        """
        # Get joint-space action from SADP_G (already 60D)
        joint_action = self._il_wrapper.get_single_step_action(il_obs)
        
        # Ensure correct shape
        if len(joint_action) != self.joint_dim:
            print(f"[WARNING] IL action has shape {joint_action.shape}, expected ({self.joint_dim},)")
            # Pad or truncate if needed
            if len(joint_action) < self.joint_dim:
                joint_action = np.pad(joint_action, (0, self.joint_dim - len(joint_action)))
            else:
                joint_action = joint_action[:self.joint_dim]
        
        return joint_action.astype(np.float32)
    
    def _normalize_pcd(self, pcd: np.ndarray) -> np.ndarray:
        """Normalize point cloud to fixed size."""
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
        stage_advanced: bool,
        final_joint_action: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict]:
        """Compute reward with stage-aware components and smoothness penalty."""
        reward = 0.0
        info = {}
        
        pcd = obs["garment_pcd"]
        current_bbox = self._compute_bbox(pcd)
        current_size = current_bbox[3:6] - current_bbox[0:3]
        initial_size = self._initial_bbox[3:6] - self._initial_bbox[0:3]
        
        # 1. Overall fold progress (from initial state)
        initial_xy = initial_size[0] * initial_size[1]
        current_xy = current_size[0] * current_size[1]
        fold_progress = (initial_xy - current_xy) / (initial_xy + 1e-6)
        fold_reward = fold_progress * self.reward_weights["fold_progress"]
        reward += fold_reward
        info["fold_progress"] = fold_progress
        
        # 2. Compactness
        initial_vol = np.prod(initial_size)
        current_vol = np.prod(current_size)
        compactness = 1.0 - current_vol / (initial_vol + 1e-6)
        reward += compactness * self.reward_weights["compactness"]
        info["compactness"] = compactness
        
        # 3. Height penalty (garment should stay flat)
        height_var = np.var(pcd[:, 2])
        reward -= height_var * self.reward_weights["height_penalty"]
        info["height_penalty"] = -height_var
        
        # 4. Residual penalty (encourage small corrections)
        residual_magnitude = np.sum(residual ** 2)
        reward -= residual_magnitude * self.reward_weights["residual_penalty"]
        info["residual_penalty"] = -residual_magnitude
        
        # 5. Smoothness penalty (penalize jerky movements)
        if final_joint_action is not None and self._last_joint_action is not None:
            action_change = np.sum((final_joint_action - self._last_joint_action) ** 2)
            reward -= action_change * self.reward_weights["smoothness_penalty"]
            info["smoothness_penalty"] = -action_change
        
        # 6. Late advance penalty (if should advance but hasn't)
        if self._il_wrapper.should_consider_stage_advance() and not stage_advanced:
            reward -= self.reward_weights["late_advance_penalty"]
            info["late_advance_penalty"] = -self.reward_weights["late_advance_penalty"]
        
        # 7. Task completion bonus
        if all(self._stages_completed) and self._check_final_success(obs):
            reward += self.reward_weights["task_success_bonus"]
            info["task_success_bonus"] = self.reward_weights["task_success_bonus"]
        
        info["total_reward"] = reward
        return reward, info
    
    def _check_final_success(self, obs: Dict[str, np.ndarray]) -> bool:
        """Check if folding task is fully complete."""
        pcd = obs["garment_pcd"]
        current_bbox = self._compute_bbox(pcd)
        current_size = current_bbox[3:6] - current_bbox[0:3]
        initial_size = self._initial_bbox[3:6] - self._initial_bbox[0:3]
        
        x_ratio = current_size[0] / (initial_size[0] + 1e-6)
        y_ratio = current_size[1] / (initial_size[1] + 1e-6)
        height_var = np.var(pcd[:, 2])
        
        return (x_ratio < 0.5) and (y_ratio < 0.7) and (height_var < 0.02)
    
    def _check_garment_anomaly(self, obs: Dict[str, np.ndarray]) -> Tuple[bool, str]:
        """
        Check if garment is in an anomalous state (stretched, lifted, out of bounds).
        
        Returns:
            (is_anomaly, anomaly_type) tuple
        """
        pcd = obs["garment_pcd"]
        
        # 1. Height anomaly: garment lifted too high (swinging)
        max_height = np.max(pcd[:, 2])
        if max_height > 0.5:  # More than 0.5m above ground
            return True, "garment_lifted"
        
        # 2. Spread anomaly: garment stretched too much
        current_bbox = self._compute_bbox(pcd)
        current_size = current_bbox[3:6] - current_bbox[0:3]
        initial_size = self._initial_bbox[3:6] - self._initial_bbox[0:3]
        
        # If XY spread increased significantly (garment being stretched)
        stretch_threshold = 1.3  # 30% larger than initial
        if current_size[0] > initial_size[0] * stretch_threshold:
            return True, "garment_stretched_x"
        if current_size[1] > initial_size[1] * stretch_threshold:
            return True, "garment_stretched_y"
        
        # 3. Position anomaly: garment center moved out of workspace
        center = np.mean(pcd, axis=0)
        if abs(center[0]) > 1.0:  # Too far left/right
            return True, "garment_out_of_bounds_x"
        if center[1] < 0.2 or center[1] > 1.5:  # Too far forward/back
            return True, "garment_out_of_bounds_y"
        
        # 4. Dispersion anomaly: garment points too spread out (being torn apart)
        point_std = np.std(pcd, axis=0)
        if np.max(point_std) > 0.5:  # Very high variance
            return True, "garment_dispersed"
        
        return False, ""
    
    def _check_early_termination(self, obs: Dict[str, np.ndarray]) -> Tuple[bool, str]:
        """
        Check conditions for early episode termination (bad exploration).
        
        Returns:
            (should_terminate, reason) tuple
        """
        # 1. Garment anomaly check (stretched, lifted, out of bounds)
        is_anomaly, anomaly_type = self._check_garment_anomaly(obs)
        if is_anomaly:
            return True, f"anomaly_{anomaly_type}"
        
        # 2. No progress for too long in current stage
        stage_steps = self._il_wrapper.stage_step_count
        stage_config = self._il_wrapper.current_stage_config
        if stage_config:
            max_allowed = stage_config.max_inference_steps * 4  # 4x normal is too long
            if stage_steps > max_allowed:
                quality = self._evaluate_current_stage_quality()
                if quality < 0.1:  # Almost no progress
                    return True, "no_stage_progress"
        
        # 3. Regression: fold quality getting worse over time
        if len(self._recent_fold_progress) >= self._progress_window:
            recent_trend = self._recent_fold_progress[-1] - self._recent_fold_progress[0]
            if recent_trend < -0.15:  # Significant regression
                return True, "regressing"
        
        # 4. Very negative fold progress (made things worse than initial)
        current_progress = self._recent_fold_progress[-1] if self._recent_fold_progress else 0
        if current_progress < -0.2:  # Garment spread out significantly
            return True, "negative_progress"
        
        return False, ""
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._env_camera.get_rgb_graph(save_or_not=False)
        return None
    
    def close(self):
        """Clean up."""
        if self._initialized:
            self._base_env.stop()
            print("[MultiStageResidualEnv] Environment closed.")


def register_multi_stage_env():
    """Register environment with Gymnasium."""
    gym.register(
        id="MultiStageResidualFoldTops-v0",
        entry_point="Env_RL.multi_stage_residual_env:MultiStageResidualEnv",
        max_episode_steps=400,
    )
