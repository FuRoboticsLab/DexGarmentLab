"""
Multi-Stage Residual RL Environment for Garment Folding with SADP_G Guidance.

This environment implements multi-stage residual RL where:
- 3 SADP_G models (frozen) provide stage-specific IL guidance
- Single RL policy learns residuals AND when to transition stages
- Stage transitions use HYBRID approach: proactive triggers + RL control

Architecture:
    action = [joint_residuals(60D), stage_advance_signal(1D)]
    final_joint_action = IL_joint_action(current_stage) + surgical_residuals
    
    NOTE: Actions are in JOINT SPACE (60D) to match SADP_G outputs exactly.
    This ensures residuals are meaningful corrections to the IL policy.

Key Features:
- Single RL policy handles all 3 stages
- Stage indicator in observation enables stage-aware behavior
- HYBRID stage transitions: proactive triggers + RL learning
- Bounded residuals ensure IL provides strong guidance
- Stage-specific rewards encourage proper sequencing
- Joint-space actions match SADP_G output format

=== HYBRID Stage Advance (NEW!) ===
Stage advances can happen in TWO ways:

1. PROACTIVE (automatic): Based on IL's natural behavior
   - When IL has run ~200 steps (its training duration)
   - When quality plateaus (no improvement for N steps)
   - RL doesn't need to learn these - they happen automatically
   
2. RL-CONTROLLED: RL can trigger early advance
   - If RL outputs advance_signal > 0.5
   - Rewards RL for learning optimal timing
   - Can discover better transitions than fixed IL duration

This hybrid approach lets RL focus on learning while benefiting from
IL's implicit knowledge about stage completion.

=== PHASE-AWARE RL Control (High-Level Decisions) ===
RL focuses on HIGH-LEVEL strategic decisions, IL handles detailed manipulation:

1. MANIPULATION PHASES:
   - APPROACH: RL CAN adjust arm position (where to grab)
   - GRASP: Pure IL (no RL intervention)
   - MANIPULATE: Pure IL (IL's expertise, RL stays out)
   - RELEASE: RL CAN adjust arm position (where to release)

2. ARM-ONLY RESIDUALS:
   - RL only adjusts arm positioning (6 joints per arm)
   - Hand/finger control is 100% IL (24 joints per hand)
   - This protects IL's grasp quality

3. PHASE DETECTION:
   - Gripper OPEN + approaching garment = APPROACH phase (RL active)
   - Gripper CLOSING or CLOSED = MANIPULATION phase (Pure IL)
   - Gripper OPENING = RELEASE phase (RL can adjust)

4. SAFE STAGE TRANSITIONS:
   - Hands automatically opened before stage advance
   - Prevents dragging cloth to new manipulation points

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
        # Logging configuration
        verbose: bool = True,          # Print training progress
        print_every_n_steps: int = 50, # Print status every N steps
        # Residual configuration - SMALLER to avoid disrupting IL
        arm_residual_scale: float = 0.05,   # Small residuals - let IL do most of the work
        hand_residual_scale: float = 0.0,   # Hand joints: ZERO - IL controls hands 100%
        action_scale: float = 0.05,
        # ========== PHASE-AWARE RL Configuration ==========
        # RL focuses on HIGH-LEVEL control (positioning), IL handles manipulation
        arm_only_residuals: bool = True,          # ONLY arm residuals, hands are pure IL
        residual_apply_interval: int = 5,         # Apply every 5 steps (sparse) - less interference
        disable_residual_during_manipulation: bool = True,   # NO residuals when gripper closed!
        gripper_threshold: float = 0.3,           # Lower threshold to detect early grasp
        # ========== Phase Detection Configuration ==========
        enable_phase_aware_control: bool = False,  # DISABLED initially - let IL work freely
        approach_phase_steps: int = 50,           # First N steps of stage = approach
        # ========== Early Termination Configuration ==========
        enable_early_termination: bool = False,   # DISABLED initially - too aggressive
        early_termination_penalty: float = 0.0,   # No penalty if disabled
        # ========== Initial State Configuration ==========
        # Match SADP_G training distribution to avoid data shift!
        use_fixed_initial_state: bool = True,     # Use fixed position (matches validation)
        fixed_pos_x: float = 0.0,                 # Fixed X position (center)
        fixed_pos_y: float = 0.8,                 # Fixed Y position
        random_pos_x_range: Tuple[float, float] = (-0.1, 0.1),  # Random X range if not fixed
        random_pos_y_range: Tuple[float, float] = (0.7, 0.9),   # Random Y range if not fixed
        # Observation configuration
        point_cloud_size: int = 2048,
        # ========== HYBRID Stage Transition Configuration ==========
        stage_advance_threshold: float = 0.5,         # RL signal threshold
        min_steps_before_advance: int = 10,           # Min steps before RL can advance
        # Proactive advance settings (based on IL behavior) - MORE LENIENT
        enable_proactive_advance: bool = True,        # Enable auto-advance triggers
        il_steps_per_stage: int = 200,                # IL's natural duration per stage
        proactive_advance_after_ratio: float = 0.85,  # Auto-advance after 85% of IL steps (was 0.9)
        min_quality_for_proactive: float = 0.1,       # Min quality for proactive advance (was 0.2, more lenient)
        quality_plateau_window: int = 30,             # Steps to detect plateau
        quality_plateau_threshold: float = 0.02,      # Max improvement to be "plateau"
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
            stage_advance_threshold: Threshold for RL's stage advance signal
            min_steps_before_advance: Minimum steps before RL can trigger advance
            enable_proactive_advance: Enable automatic stage advances based on IL behavior
            il_steps_per_stage: IL's natural duration per stage (SADP_G uses ~200)
            proactive_advance_after_ratio: Auto-advance after this ratio of IL steps
            min_quality_for_proactive: Minimum fold quality for proactive advance
            quality_plateau_window: Window size to detect quality plateau
            quality_plateau_threshold: Max improvement to be considered "plateau"
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
        
        # ========== HYBRID Stage Advance Configuration ==========
        self.enable_proactive_advance = enable_proactive_advance
        self.il_steps_per_stage = il_steps_per_stage
        self.proactive_advance_after_ratio = proactive_advance_after_ratio
        self.min_quality_for_proactive = min_quality_for_proactive
        self.quality_plateau_window = quality_plateau_window
        self.quality_plateau_threshold = quality_plateau_threshold
        
        # ========== Phase-Aware RL Control ==========
        self.enable_phase_aware_control = enable_phase_aware_control
        self.approach_phase_steps = approach_phase_steps
        self._current_phase = "approach"  # Track current manipulation phase
        
        # ========== Early Termination Configuration ==========
        self.enable_early_termination = enable_early_termination
        self.early_termination_penalty = early_termination_penalty
        
        # ========== Logging Configuration ==========
        self.verbose = verbose
        self.print_every_n_steps = print_every_n_steps
        self._episode_count = 0
        self._total_rl_advances = 0
        self._total_proactive_advances = 0
        
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
        
        # ========== HYBRID Stage Advance Tracking ==========
        self._stage_quality_history = []  # Track quality within current stage
        self._proactive_advance_triggered = False  # Flag for logging
        
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
            # NEW: Hint for RL about when to advance (helps faster learning)
            "should_advance_hint": spaces.Box(
                low=0, high=1,
                shape=(1,),  # 1.0 = proactive conditions met, 0.0 = not yet
                dtype=np.float32
            ),
            # NEW: Current manipulation phase (helps RL know when it can act)
            "manipulation_phase": spaces.Box(
                low=0, high=1,
                shape=(3,),  # One-hot: [approach, manipulation, release]
                dtype=np.float32
            ),
        })
        
        # Reward weights (ENHANCED: More positive, less negative to encourage learning)
        self.reward_weights = {
            # Task rewards (increased to encourage progress)
            "fold_progress": 2.0,      # Increased from 1.0
            "compactness": 1.0,         # Increased from 0.5
            "height_penalty": 0.2,     # Reduced from 0.3 (less harsh)
            # Residual rewards (reduced penalties)
            "residual_penalty": 0.01,   # Reduced from 0.02 (allow more exploration)
            "smoothness_penalty": 0.02, # Reduced from 0.05 (less harsh)
            # Stage transition rewards (increased bonuses)
            "stage_advance_bonus": 10.0,      # Increased from 5.0
            "premature_advance_penalty": 1.0, # Reduced from 2.0 (less harsh)
            "late_advance_penalty": 0.05,    # Reduced from 0.1
            # Completion rewards (increased to encourage success)
            "stage_completion_bonus": 5.0,   # Increased from 3.0
            "task_success_bonus": 30.0,      # Increased from 20.0
            # Early termination penalty (disabled if early termination is off)
            "early_termination_penalty": 0.0, # Set to 0 (handled by flag)
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
        self._stage_quality_history = []
        self._proactive_advance_triggered = False
        
        # Reset episode-level counters
        self._episode_count += 1
        self._total_rl_advances = 0
        self._total_proactive_advances = 0
        self._current_phase = "approach"  # Start in approach phase
        
        if self.verbose:
            print(f"\n🎬 Starting Episode {self._episode_count}")
            print(f"   IL Policy: {'Dummy' if self.use_dummy_il else 'SADP_G'}")
            print(f"   Phase-Aware Control: {'Enabled' if self.enable_phase_aware_control else 'Disabled'}")
            print(f"   Early Termination: {'Enabled' if self.enable_early_termination else 'Disabled'}")
            print(f"   Proactive Advance: {'Enabled' if self.enable_proactive_advance else 'Disabled'}")
            print(f"   Residual Scale: Arm={self.arm_residual_scale:.3f}, Hand={self.hand_residual_scale:.3f}")
            print(f"   Residual Interval: Every {self.residual_apply_interval} steps")
        
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
        
        # Update garment state BEFORE stage advance checks
        self._update_garment_pcd_and_gam()
        
        # Track stage quality for plateau detection
        stage_quality = self._evaluate_current_stage_quality()
        self._stage_quality_history.append(stage_quality)
        
        # ========== HYBRID Stage Advance ==========
        stage_advanced = False
        advance_reward = 0.0
        advance_source = None  # "proactive" or "rl" for logging
        self._proactive_advance_triggered = False
        
        # Priority 1: Check PROACTIVE advance (automatic based on IL behavior)
        if self._check_proactive_advance():
            stage_advanced, advance_reward = self._execute_proactive_advance()
            if stage_advanced:
                advance_source = "proactive"
                self._total_proactive_advances += 1
                if self.verbose:
                    current_stage = self._il_wrapper.current_stage
                    stage_name = current_stage.name if hasattr(current_stage, 'name') else str(current_stage)
                    print(f"  📦 [PROACTIVE] Stage advanced from {stage_name}!")
                    print(f"     Quality: {stage_quality:.3f}, Steps: {self._il_wrapper.stage_step_count}/{self.il_steps_per_stage}")
        
        # Priority 2: RL-initiated advance (if proactive didn't trigger)
        if not stage_advanced and stage_advance_signal > self.stage_advance_threshold:
            stage_advanced, advance_reward = self._handle_stage_advance()
            if stage_advanced:
                advance_source = "rl"
                self._total_rl_advances += 1
                # Bonus for RL learning good timing!
                advance_reward += 2.0  # Extra reward for RL-initiated advance
                if self.verbose:
                    print(f"  🤖 [RL] Stage advanced! Quality: {stage_quality:.2f}, Steps in stage: {self._il_wrapper.stage_step_count}")
        
        # Reset quality history if stage advanced
        if stage_advanced:
            self._stage_quality_history = []
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward (includes smoothness penalty)
        # Use effective_residual for reward so we only penalize residuals that were actually applied
        reward, reward_info = self._compute_reward(obs, effective_residual, stage_advanced, final_joint_action)
        reward += advance_reward
        reward_info["advance_reward"] = advance_reward
        reward_info["residual_applied"] = apply_residual
        reward_info["effective_residual_norm"] = np.linalg.norm(effective_residual)
        reward_info["advance_source"] = advance_source
        reward_info["stage_quality"] = stage_quality
        reward_info["should_advance_hint"] = self._get_should_advance_hint()
        reward_info["manipulation_phase"] = self._current_phase
        
        # Track progress for early termination
        self._recent_fold_progress.append(reward_info.get("fold_progress", 0))
        if len(self._recent_fold_progress) > self._progress_window:
            self._recent_fold_progress.pop(0)
        
        # Check for early termination (bad exploration) - ONLY if enabled
        early_term = False
        term_reason = ""
        if self.enable_early_termination:
            early_term, term_reason = self._check_early_termination(obs)
        
        # Check termination
        all_stages_done = all(self._stages_completed)
        terminated = all_stages_done and self._check_final_success(obs)
        
        # Early termination overrides normal termination (only if enabled)
        if early_term and self.enable_early_termination:
            terminated = True
            reward -= self.early_termination_penalty
            reward_info["early_termination"] = True
            reward_info["termination_reason"] = term_reason
            if self.verbose:
                print(f"  ⚠️ Early termination: {term_reason}")
        
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
        
        # ========== Verbose Logging ==========
        if self.verbose:
            # Periodic status print (ENHANCED with more info)
            if self.current_step % self.print_every_n_steps == 0:
                current_stage = self._il_wrapper.current_stage
                stage_name = current_stage.name if hasattr(current_stage, 'name') else str(current_stage)
                stages_done = sum(self._stages_completed)
                hint = self._get_should_advance_hint()
                phase_emoji = {"approach": "🎯", "manipulation": "🤲", "release": "🖐️"}.get(self._current_phase, "❓")
                rl_status = "RL✓" if apply_residual else "IL"
                stage_steps = self._il_wrapper.stage_step_count
                progress_pct = (stage_steps / self.il_steps_per_stage) * 100
                print(f"  [Step {self.current_step:3d}] Stage: {stage_name} ({stage_steps}/{self.il_steps_per_stage}, {progress_pct:.0f}%) | "
                      f"Phase: {phase_emoji}{self._current_phase:12s} | {rl_status} | "
                      f"Quality: {stage_quality:.3f} | Hint: {hint:.2f} | Stages: {stages_done}/3 | Reward: {reward:.2f}")
            
            # Episode end summary
            if terminated or truncated:
                self._print_episode_summary(info, terminated, truncated, early_term if early_term else False)
        
        return obs, reward, terminated, truncated, info
    
    def _print_episode_summary(self, info: Dict, terminated: bool, truncated: bool, early_term: bool):
        """Print episode summary at the end of each episode."""
        is_success = info.get("is_success", False)
        stages_completed = sum(self._stages_completed)
        final_quality = info.get("stage_quality", 0)
        total_reward = info.get("total_reward", 0)
        
        # Get current stage info
        current_stage = self._il_wrapper.current_stage
        stage_name = current_stage.name if hasattr(current_stage, 'name') else str(current_stage)
        stage_steps = self._il_wrapper.stage_step_count
        
        # Determine outcome
        if is_success:
            outcome = "✅ SUCCESS"
        elif early_term:
            reason = info.get("termination_reason", "unknown")
            outcome = f"❌ EARLY TERMINATION ({reason})"
        elif truncated:
            outcome = "⏱️ TRUNCATED (max steps)"
        else:
            outcome = "❌ FAILED"
        
        print(f"\n{'='*60}")
        print(f"📊 Episode {self._episode_count} Summary")
        print(f"{'='*60}")
        print(f"  Outcome:        {outcome}")
        print(f"  Steps:          {self.current_step}")
        print(f"  Current Stage:  {stage_name} (step {stage_steps}/{self.il_steps_per_stage})")
        print(f"  Stages Done:    {stages_completed}/3 {self._stages_completed}")
        print(f"  Final Quality:  {final_quality:.3f}")
        print(f"  Total Reward:   {total_reward:.3f}")
        print(f"  RL Advances:    {self._total_rl_advances} (this episode)")
        print(f"  Auto Advances:  {self._total_proactive_advances} (this episode)")
        if stages_completed == 0:
            print(f"  ⚠️  WARNING: No stages completed! Check IL execution.")
        print(f"{'='*60}\n")
    
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
        
        PHASE-AWARE CONTROL:
        - APPROACH phase: RL CAN adjust arm positioning (where to grab)
        - MANIPULATION phase: Pure IL (gripper closed, moving cloth)
        - RELEASE phase: RL CAN adjust (gripper opening)
        
        This lets RL make HIGH-LEVEL decisions while IL handles detailed manipulation.
        
        Returns:
            True if residual should be applied
        """
        # Check 1: Sparse application - only every N steps
        if self.current_step % self.residual_apply_interval != 0:
            return False
        
        # Check 2: Phase-aware gating
        if self.enable_phase_aware_control:
            phase = self._detect_manipulation_phase()
            self._current_phase = phase  # Store for logging
            
            # Only allow residuals in approach and release phases
            if phase == "manipulation":
                return False  # Pure IL during manipulation
            # "approach" and "release" phases allow residuals
        
        # Fallback: Original gripper-based check
        elif self.disable_residual_during_manipulation:
            if self._is_gripper_closed():
                return False
        
        return True
    
    def _detect_manipulation_phase(self) -> str:
        """
        Detect current manipulation phase within the stage.
        
        Phases:
        - "approach": Moving toward grasp point (gripper open, early in stage)
        - "manipulation": Grasping and moving cloth (gripper closed)
        - "release": Releasing cloth (gripper opening, late in stage)
        
        Returns:
            Phase name: "approach", "manipulation", or "release"
        """
        stage_steps = self._il_wrapper.stage_step_count
        gripper_closed = self._is_gripper_closed()
        
        # Early in stage + gripper open = approach phase
        if stage_steps < self.approach_phase_steps and not gripper_closed:
            return "approach"
        
        # Gripper closed = manipulation phase (IL controls)
        if gripper_closed:
            return "manipulation"
        
        # Late in stage + gripper open = release phase
        # (After manipulation, hand opened for release)
        if stage_steps > self.approach_phase_steps and not gripper_closed:
            return "release"
        
        # Default to manipulation (safest)
        return "manipulation"
    
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
        Handle RL-initiated stage advance attempt.
        
        IMPORTANT: Opens hands before advancing to prevent dragging cloth!
        
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
        # RELAXED: Lower threshold to allow more advances (was 0.3, now 0.15)
        stage_quality = self._evaluate_current_stage_quality()
        
        if stage_quality < 0.15:
            # Too early to advance (but more lenient)
            return False, -self.reward_weights["premature_advance_penalty"]
        
        # ========== SAFE TRANSITION: Open hands first! ==========
        # This prevents dragging the cloth to the new manipulation point
        self._safe_release_before_transition()
        
        # Mark current stage as completed
        stage_idx = int(current_stage) - 1
        self._stages_completed[stage_idx] = True
        
        # Advance to next stage
        success = self._il_wrapper.advance_stage()
        
        if success:
            # Reset stage quality history for new stage
            self._stage_quality_history = []
            
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
    
    def _safe_release_before_transition(self):
        """
        Safely release any grasped cloth before transitioning to next stage.
        
        This is CRITICAL to prevent dragging cloth when moving to new manipulation point.
        
        Steps:
        1. Open both hands
        2. Let cloth settle with increased gravity
        3. Move arms away from cloth
        """
        if self.verbose:
            print("  🖐️ Safe release: Opening hands before stage transition...")
        
        # 1. Open both hands
        self._robot.set_both_hand_state("open", "open")
        for _ in range(30):
            self._base_env.step()
        
        # 2. Let cloth settle with increased gravity
        self._garment.particle_material.set_gravity_scale(10.0)
        for _ in range(50):
            self._base_env.step()
        self._garment.particle_material.set_gravity_scale(1.0)
        
        # 3. Move arms up and away from cloth to avoid collision during repositioning
        try:
            # Move left arm up
            left_pos, _ = self._robot.dexleft.get_cur_ee_pos()
            left_target = np.array([left_pos[0], left_pos[1], 0.4])  # Lift up
            self._robot.dexleft.dense_step_action(
                target_pos=left_target,
                target_ori=np.array([0.579, -0.579, -0.406, 0.406]),
                angular_type="quat"
            )
            
            # Move right arm up
            right_pos, _ = self._robot.dexright.get_cur_ee_pos()
            right_target = np.array([right_pos[0], right_pos[1], 0.4])  # Lift up
            self._robot.dexright.dense_step_action(
                target_pos=right_target,
                target_ori=np.array([0.406, -0.406, -0.579, 0.579]),
                angular_type="quat"
            )
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️ Warning: Could not lift arms: {e}")
        
        # Final settle
        for _ in range(20):
            self._base_env.step()
    
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
    
    # ==================== HYBRID Stage Advance Methods ====================
    
    def _check_proactive_advance(self) -> bool:
        """
        Check if proactive (automatic) stage advance should happen.
        
        Proactive advance triggers when:
        1. IL has run most of its natural duration (~90% of steps)
        2. Quality is above minimum threshold
        3. OR quality has plateaued (no improvement for N steps)
        
        Returns:
            True if proactive advance should happen
        """
        if not self.enable_proactive_advance:
            return False
        
        current_stage = self._il_wrapper.current_stage
        if current_stage == FoldingStage.COMPLETED:
            return False
        
        stage_steps = self._il_wrapper.stage_step_count
        stage_quality = self._evaluate_current_stage_quality()
        
        # Condition 1: IL has run most of its natural duration
        steps_threshold = int(self.il_steps_per_stage * self.proactive_advance_after_ratio)
        duration_met = stage_steps >= steps_threshold
        quality_ok = stage_quality >= self.min_quality_for_proactive
        
        # More lenient: If duration met, advance even with low quality (IL might be done)
        if duration_met:
            # If quality is decent, definitely advance
            if quality_ok:
                return True
            # Even with low quality, advance if we've run long enough (IL might be stuck)
            if stage_steps >= self.il_steps_per_stage:  # Full duration
                return True
        
        # Condition 2: Quality has plateaued (even if duration not fully met)
        if self._detect_quality_plateau() and stage_quality >= self.min_quality_for_proactive:
            return True
        
        return False
    
    def _detect_quality_plateau(self) -> bool:
        """
        Detect if stage quality has plateaued (no improvement).
        
        This indicates IL has done what it can and it's time to move on.
        
        Returns:
            True if quality has plateaued
        """
        if len(self._stage_quality_history) < self.quality_plateau_window:
            return False
        
        recent = self._stage_quality_history[-self.quality_plateau_window:]
        improvement = recent[-1] - recent[0]
        
        # If quality hasn't improved much, it's plateaued
        return improvement < self.quality_plateau_threshold
    
    def _get_should_advance_hint(self) -> float:
        """
        Get hint value for observation (helps RL learn faster).
        
        Returns value in [0, 1]:
        - 0.0: Not ready to advance
        - 0.5: Getting close (>50% of IL steps)
        - 1.0: Proactive conditions fully met
        """
        current_stage = self._il_wrapper.current_stage
        if current_stage == FoldingStage.COMPLETED:
            return 0.0
        
        stage_steps = self._il_wrapper.stage_step_count
        stage_quality = self._evaluate_current_stage_quality()
        
        # Base hint from step progress
        step_ratio = stage_steps / self.il_steps_per_stage
        step_hint = np.clip(step_ratio, 0.0, 1.0)
        
        # Quality contribution
        quality_hint = np.clip(stage_quality / 0.3, 0.0, 1.0)  # 0.3 = good quality
        
        # Combined hint (both duration and quality matter)
        combined = (step_hint + quality_hint) / 2.0
        
        # Boost to 1.0 if proactive conditions fully met
        if self._check_proactive_advance():
            combined = 1.0
        
        return combined
    
    def _execute_proactive_advance(self) -> Tuple[bool, float]:
        """
        Execute a proactive (automatic) stage advance.
        
        Similar to _handle_stage_advance but with different reward structure
        since this is automatic, not RL-initiated.
        
        IMPORTANT: Opens hands before advancing to prevent dragging cloth!
        
        Returns:
            (success, reward) tuple
        """
        current_stage = self._il_wrapper.current_stage
        stage_quality = self._evaluate_current_stage_quality()
        
        # ========== SAFE TRANSITION: Open hands first! ==========
        # This prevents dragging the cloth to the new manipulation point
        self._safe_release_before_transition()
        
        # Mark current stage as completed
        stage_idx = int(current_stage) - 1
        if 0 <= stage_idx < 3:
            self._stages_completed[stage_idx] = True
        
        # Advance to next stage
        success = self._il_wrapper.advance_stage()
        
        if success:
            # Reset stage quality history for new stage
            self._stage_quality_history = []
            
            # Store new stage initial bbox
            self._stage_initial_bbox = self._compute_bbox(self._current_garment_pcd)
            
            # Update garment state and GAM for new stage
            self._update_garment_pcd_and_gam()
            
            # Pre-position robot for new stage
            self._pre_position_robot_for_stage()
            
            # Reward for completing stage (smaller than RL-initiated to encourage RL learning)
            advance_reward = self.reward_weights["stage_completion_bonus"]
            
            self._proactive_advance_triggered = True
            return True, advance_reward
        
        return False, 0.0
    
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
        
        # ========== NaN Protection for Point Cloud ==========
        if np.isnan(self._current_garment_pcd).any() or np.isinf(self._current_garment_pcd).any():
            if self.verbose:
                print("  ⚠️ WARNING: Invalid values in point cloud, replacing with zeros")
            self._current_garment_pcd = np.nan_to_num(
                self._current_garment_pcd, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
        
        # Get GAM keypoints and similarity
        try:
            keypoints, indices, similarity = self._gam_model.get_manipulation_points(
                input_pcd=self._current_garment_pcd,
                index_list=[957, 501, 1902, 448, 1196, 422]
            )
            self._gam_keypoints = keypoints.astype(np.float32)
            self._gam_similarity = similarity
            
            # NaN protection for keypoints
            if np.isnan(self._gam_keypoints).any():
                if self.verbose:
                    print("  ⚠️ WARNING: NaN in GAM keypoints, replacing with zeros")
                self._gam_keypoints = np.nan_to_num(self._gam_keypoints, nan=0.0).astype(np.float32)
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️ WARNING: GAM failed: {e}")
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
        
        # Get current manipulation phase one-hot encoding
        phase = self._detect_manipulation_phase() if self.enable_phase_aware_control else "manipulation"
        phase_one_hot = np.zeros(3, dtype=np.float32)
        if phase == "approach":
            phase_one_hot[0] = 1.0
        elif phase == "manipulation":
            phase_one_hot[1] = 1.0
        elif phase == "release":
            phase_one_hot[2] = 1.0
        
        obs = {
            "garment_pcd": self._current_garment_pcd.astype(np.float32),
            "joint_positions": joint_positions,
            "ee_poses": ee_poses,
            "il_action": il_action,  # Now 60D joint space
            "gam_keypoints": self._gam_keypoints,
            "current_stage": self._il_wrapper.get_stage_one_hot(),
            "stage_progress": np.array([self._il_wrapper.get_stage_progress()], dtype=np.float32),
            "stages_completed": np.array(self._stages_completed, dtype=np.float32),
            # Hint for RL about when to advance
            "should_advance_hint": np.array([self._get_should_advance_hint()], dtype=np.float32),
            # Current manipulation phase [approach, manipulation, release]
            "manipulation_phase": phase_one_hot,
        }
        
        # ========== NaN Protection ==========
        # Replace any NaN/Inf values to prevent policy network crash
        for key, value in obs.items():
            if np.isnan(value).any() or np.isinf(value).any():
                if self.verbose:
                    nan_count = np.isnan(value).sum()
                    inf_count = np.isinf(value).sum()
                    print(f"  ⚠️ WARNING: NaN/Inf in '{key}' (NaN: {nan_count}, Inf: {inf_count})")
                obs[key] = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        return obs
    
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
            if self.verbose:
                print(f"  ⚠️ WARNING: IL action has shape {joint_action.shape}, expected ({self.joint_dim},)")
            # Pad or truncate if needed
            if len(joint_action) < self.joint_dim:
                joint_action = np.pad(joint_action, (0, self.joint_dim - len(joint_action)))
            else:
                joint_action = joint_action[:self.joint_dim]
        
        # ========== NaN Protection for IL Action ==========
        if np.isnan(joint_action).any() or np.isinf(joint_action).any():
            if self.verbose:
                print("  ⚠️ WARNING: NaN/Inf in IL action, using current joint positions")
            # Fallback to current joint positions (stay in place)
            left_joints = self._robot.dexleft.get_joint_positions()
            right_joints = self._robot.dexright.get_joint_positions()
            joint_action = np.concatenate([left_joints, right_joints])
        
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
        
        RELAXED THRESHOLDS: Only trigger on extreme cases to allow IL to work.
        
        Returns:
            (is_anomaly, anomaly_type) tuple
        """
        pcd = obs["garment_pcd"]
        
        if len(pcd) == 0:
            return False, ""  # Empty point cloud is handled elsewhere
        
        # 1. Height anomaly: garment lifted too high (swinging)
        # RELAXED: Only trigger if extremely high (was 0.5m, now 1.0m)
        max_height = np.max(pcd[:, 2])
        if max_height > 1.0:  # More than 1.0m above ground (very extreme)
            return True, "garment_lifted"
        
        # 2. Spread anomaly: garment stretched too much
        current_bbox = self._compute_bbox(pcd)
        current_size = current_bbox[3:6] - current_bbox[0:3]
        initial_size = self._initial_bbox[3:6] - self._initial_bbox[0:3]
        
        # RELAXED: Only trigger if extremely stretched (was 1.3x, now 2.0x)
        stretch_threshold = 2.0  # 100% larger than initial (very extreme)
        if current_size[0] > initial_size[0] * stretch_threshold:
            return True, "garment_stretched_x"
        if current_size[1] > initial_size[1] * stretch_threshold:
            return True, "garment_stretched_y"
        
        # 3. Position anomaly: garment center moved out of workspace
        # RELAXED: Only trigger if way out of bounds (was 1.0m, now 2.0m)
        center = np.mean(pcd, axis=0)
        if abs(center[0]) > 2.0:  # Way too far left/right
            return True, "garment_out_of_bounds_x"
        if center[1] < -0.5 or center[1] > 2.5:  # Way too far forward/back
            return True, "garment_out_of_bounds_y"
        
        # 4. Dispersion anomaly: garment points too spread out (being torn apart)
        # RELAXED: Only trigger on extreme dispersion (was 0.5, now 1.0)
        point_std = np.std(pcd, axis=0)
        if np.max(point_std) > 1.0:  # Very extreme variance
            return True, "garment_dispersed"
        
        return False, ""
    
    def _check_early_termination(self, obs: Dict[str, np.ndarray]) -> Tuple[bool, str]:
        """
        Check conditions for early episode termination (bad exploration).
        
        VERY LENIENT: Only terminate on extreme cases to allow IL to work.
        Most checks are disabled or have very relaxed thresholds.
        
        Returns:
            (should_terminate, reason) tuple
        """
        # 1. Garment anomaly check (stretched, lifted, out of bounds)
        # Only check extreme anomalies
        is_anomaly, anomaly_type = self._check_garment_anomaly(obs)
        if is_anomaly:
            return True, f"anomaly_{anomaly_type}"
        
        # 2. No progress for too long in current stage
        # RELAXED: Only check if way beyond normal (was 4x, now 10x)
        stage_steps = self._il_wrapper.stage_step_count
        stage_config = self._il_wrapper.current_stage_config
        if stage_config:
            max_allowed = stage_config.max_inference_steps * 10  # 10x normal (very lenient)
            if stage_steps > max_allowed:
                quality = self._evaluate_current_stage_quality()
                # Only terminate if absolutely no progress (was 0.1, now 0.0)
                if quality < 0.0:  # Negative quality (impossible, but safe check)
                    return True, "no_stage_progress"
        
        # 3. Regression: fold quality getting worse over time
        # DISABLED: Too aggressive, IL might have temporary regressions
        # if len(self._recent_fold_progress) >= self._progress_window:
        #     recent_trend = self._recent_fold_progress[-1] - self._recent_fold_progress[0]
        #     if recent_trend < -0.15:  # Significant regression
        #         return True, "regressing"
        
        # 4. Very negative fold progress (made things worse than initial)
        # RELAXED: Only trigger on extreme negative progress (was -0.2, now -0.5)
        current_progress = self._recent_fold_progress[-1] if self._recent_fold_progress else 0
        if current_progress < -0.5:  # Garment spread out extremely (very rare)
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
