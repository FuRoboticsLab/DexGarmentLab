"""
SIMPLIFIED Multi-Stage Residual RL Environment for Garment Folding.

This is a simplified version that removes complex mechanisms to focus on core learning:
- Removed: Phase-aware control, early termination, complex gating
- Kept: Multi-stage IL, proactive advance, simple residuals

Architecture:
    action = [joint_residuals(60D), stage_advance_signal(1D)]
    final_joint_action = IL_joint_action(current_stage) + arm_residuals
    
Key Simplifications:
1. Arm-only residuals (hands are 100% IL-controlled)
2. Residuals applied every step (no complex gating)
3. No early termination (let IL complete stages)
4. Simple reward function (fewer components)
5. Proactive advance helps IL complete stages

Usage:
    env = MultiStageResidualEnvSimple(
        training_data_num=100,
        stage_checkpoints=[1500, 1500, 1500],
    )
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Env_RL.multi_stage_sadpg_wrapper import (
    create_multi_stage_wrapper,
    FoldingStage,
    STAGE_CONFIGS,
)


class MultiStageResidualEnvSimple(gym.Env):
    """
    SIMPLIFIED Multi-Stage Residual RL Environment.
    
    Action Space (61D continuous):
        [0:30]  left arm joint residuals (6 arm DOF)
        [30:60] right arm joint residuals (6 arm DOF)
        [60]    stage_advance_signal (>0.5 = attempt advance)
        Note: Hand joints are ZERO - IL controls hands 100%
        
    Observation Space (Dict):
        - garment_pcd: (2048, 3) garment point cloud
        - joint_positions: (60,) current joint positions
        - ee_poses: (14,) end-effector poses
        - il_action: (60,) proposed joint action from current stage's SADP_G
        - gam_keypoints: (6, 3) manipulation keypoints
        - current_stage: (4,) one-hot [stage1, stage2, stage3, completed]
        - stage_progress: (1,) normalized progress in current stage
        - stages_completed: (3,) binary mask of completed stages
        - should_advance_hint: (1,) hint for RL about when to advance
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
        max_episode_steps: int = 400,
        # Residual configuration (SIMPLIFIED)
        arm_residual_scale: float = 0.05,   # Small residuals - don't disrupt IL
        # Proactive advance (helps IL complete stages)
        enable_proactive_advance: bool = True,
        il_steps_per_stage: int = 200,
        proactive_advance_after_ratio: float = 0.85,
        min_quality_for_proactive: float = 0.1,
        # Observation configuration
        point_cloud_size: int = 2048,
        # Stage transition
        stage_advance_threshold: float = 0.5,
        min_steps_before_advance: int = 10,
        # Logging
        verbose: bool = True,
        print_every_n_steps: int = 50,
        # Device
        device: str = "cuda:0",
    ):
        super().__init__()
        
        # Store configuration
        self.training_data_num = training_data_num
        self.stage_checkpoints = [stage_1_checkpoint, stage_2_checkpoint, stage_3_checkpoint]
        self.use_dummy_il = use_dummy_il
        self.config = config or {}
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.arm_residual_scale = arm_residual_scale
        self.point_cloud_size = point_cloud_size
        self.device = device
        self.verbose = verbose
        self.print_every_n_steps = print_every_n_steps
        
        # Proactive advance settings
        self.enable_proactive_advance = enable_proactive_advance
        self.il_steps_per_stage = il_steps_per_stage
        self.proactive_advance_after_ratio = proactive_advance_after_ratio
        self.min_quality_for_proactive = min_quality_for_proactive
        self.stage_advance_threshold = stage_advance_threshold
        self.min_steps_before_advance = min_steps_before_advance
        
        # Will be initialized lazily
        self._initialized = False
        self._il_wrapper = None
        
        # Episode state
        self.current_step = 0
        self._initial_bbox = None
        self._stage_initial_bbox = None
        self._last_il_action = None
        self._stages_completed = [False, False, False]
        self._stage_quality_history = []
        self._episode_count = 0
        self._total_rl_advances = 0
        self._total_proactive_advances = 0
        
        # Joint space dimensions
        self.joint_dim = 60
        self.arm_dof = 6   # UR10e arm joints
        self.hand_dof = 24 # Shadow hand joints
        
        # Action space: arm_residuals(12D) + hand_zeros(48D) + stage_advance(1D) = 61D
        # But we only apply residuals to arms, hands are always zero
        low_bounds = np.concatenate([
            [-arm_residual_scale] * self.arm_dof,    # Left arm
            [0.0] * self.hand_dof,                    # Left hand (always zero)
            [-arm_residual_scale] * self.arm_dof,    # Right arm
            [0.0] * self.hand_dof,                    # Right hand (always zero)
            [-1.0]  # Stage advance signal
        ])
        high_bounds = np.concatenate([
            [arm_residual_scale] * self.arm_dof,
            [0.0] * self.hand_dof,
            [arm_residual_scale] * self.arm_dof,
            [0.0] * self.hand_dof,
            [1.0]
        ])
        
        self.action_space = spaces.Box(
            low=low_bounds.astype(np.float32),
            high=high_bounds.astype(np.float32),
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
                shape=(self.joint_dim,),
                dtype=np.float32
            ),
            "ee_poses": spaces.Box(
                low=-10.0, high=10.0,
                shape=(14,),
                dtype=np.float32
            ),
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
                shape=(4,),
                dtype=np.float32
            ),
            "stage_progress": spaces.Box(
                low=0, high=1,
                shape=(1,),
                dtype=np.float32
            ),
            "stages_completed": spaces.Box(
                low=0, high=1,
                shape=(3,),
                dtype=np.float32
            ),
            "should_advance_hint": spaces.Box(
                low=0, high=1,
                shape=(1,),
                dtype=np.float32
            ),
        })
        
        # SIMPLIFIED Reward weights
        self.reward_weights = {
            "fold_progress": 2.0,
            "compactness": 1.0,
            "height_penalty": 0.2,
            "residual_penalty": 0.01,
            "stage_advance_bonus": 10.0,
            "premature_advance_penalty": 1.0,
            "stage_completion_bonus": 5.0,
            "task_success_bonus": 30.0,
        }
        
    def _lazy_init(self):
        """Initialize Isaac Sim environment and IL policy."""
        if self._initialized:
            return
        
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
        
        if self.verbose:
            print("[MultiStageResidualEnvSimple] Initializing environment...")
        
        # Create base environment
        self._base_env = BaseEnv()
        
        # Add ground
        self._ground = Real_Ground(
            self._base_env.scene,
            visual_material_usd=self.config.get("ground_material_usd"),
        )
        
        # Add garment
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
        self._il_wrapper = create_multi_stage_wrapper(
            use_dummy=self.use_dummy_il,
            training_data_num=self.training_data_num,
            stage_1_checkpoint=self.stage_checkpoints[0],
            stage_2_checkpoint=self.stage_checkpoints[1],
            stage_3_checkpoint=self.stage_checkpoints[2],
            device=self.device,
            lazy_load=False,
        )
        
        # Cache for observations
        self._current_garment_pcd = None
        self._gam_keypoints = None
        self._current_affordance = None
        
        self._initialized = True
        if self.verbose:
            print("[MultiStageResidualEnvSimple] Environment initialized!")
    
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
        self._last_il_action = None
        self._stage_quality_history = []
        self._episode_count += 1
        self._total_rl_advances = 0
        self._total_proactive_advances = 0
        
        # Full world reset
        self._base_env.reset()
        
        # Fixed garment position
        if seed is not None:
            np.random.seed(seed)
        
        pos = np.array([0.0, 0.8, 0.2])
        ori = np.array([0.0, 0.0, 0.0])
        
        self._garment.set_pose(pos=pos, ori=ori)
        
        # Reset robot
        self._robot.dexleft.post_reset()
        self._robot.dexright.post_reset()
        self._robot.set_both_hand_state("open", "open")
        
        # Settle
        self._garment.particle_material.set_gravity_scale(10.0)
        for _ in range(50):
            self._base_env.step()
        self._garment.particle_material.set_gravity_scale(1.0)
        for _ in range(100):
            self._base_env.step()
        
        # Get initial state
        self._update_garment_pcd_and_gam()
        self._initial_bbox = self._compute_bbox(self._current_garment_pcd)
        self._stage_initial_bbox = self._initial_bbox.copy()
        
        # Pre-position robot
        self._pre_position_robot_for_stage()
        
        obs = self._get_observation()
        
        if self.verbose:
            print(f"\n🎬 Starting Episode {self._episode_count} (SIMPLIFIED)")
            print(f"   IL Policy: {'Dummy' if self.use_dummy_il else 'SADP_G'}")
            print(f"   Proactive Advance: {'Enabled' if self.enable_proactive_advance else 'Disabled'}")
        
        return obs, {"initial_pos": pos}
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute step with simplified residual action.
        
        SIMPLIFIED: Just apply arm residuals, no complex gating.
        """
        self.current_step += 1
        
        # Parse action
        residual_action = action[:self.joint_dim]
        stage_advance_signal = action[self.joint_dim]
        
        # Get IL action
        il_obs = self._get_il_observation()
        il_action = self._get_il_joint_action(il_obs)
        self._last_il_action = il_action
        
        # SIMPLIFIED: Apply arm-only residuals (hands always zero)
        arm_residual = self._get_arm_only_residual(residual_action)
        final_joint_action = il_action + arm_residual
        
        # Execute
        self._execute_joint_action(final_joint_action)
        
        # Update IL observation history
        self._il_wrapper.update_obs(il_obs)
        
        # Update garment state
        self._update_garment_pcd_and_gam()
        
        # Track quality
        stage_quality = self._evaluate_current_stage_quality()
        self._stage_quality_history.append(stage_quality)
        
        # Handle stage advance (SIMPLIFIED: just proactive + RL)
        stage_advanced = False
        advance_reward = 0.0
        advance_source = None
        
        # Priority 1: Proactive advance
        if self._check_proactive_advance():
            stage_advanced, advance_reward = self._execute_proactive_advance()
            if stage_advanced:
                advance_source = "proactive"
                self._total_proactive_advances += 1
                if self.verbose:
                    print(f"  📦 [PROACTIVE] Stage advanced! Quality: {stage_quality:.3f}")
        
        # Priority 2: RL-initiated advance
        if not stage_advanced and stage_advance_signal > self.stage_advance_threshold:
            stage_advanced, advance_reward = self._handle_stage_advance()
            if stage_advanced:
                advance_source = "rl"
                self._total_rl_advances += 1
                advance_reward += 2.0  # Bonus for RL learning
                if self.verbose:
                    print(f"  🤖 [RL] Stage advanced! Quality: {stage_quality:.3f}")
        
        if stage_advanced:
            self._stage_quality_history = []
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward (SIMPLIFIED)
        reward, reward_info = self._compute_reward(obs, arm_residual, stage_advanced)
        reward += advance_reward
        reward_info["advance_reward"] = advance_reward
        reward_info["advance_source"] = advance_source
        reward_info["stage_quality"] = stage_quality
        
        # Check termination
        all_stages_done = all(self._stages_completed)
        terminated = all_stages_done and self._check_final_success(obs)
        truncated = self.current_step >= self.max_episode_steps
        
        # Logging
        if self.verbose:
            if self.current_step % self.print_every_n_steps == 0:
                current_stage = self._il_wrapper.current_stage
                stage_name = current_stage.name if hasattr(current_stage, 'name') else str(current_stage)
                stages_done = sum(self._stages_completed)
                stage_steps = self._il_wrapper.stage_step_count
                progress_pct = (stage_steps / self.il_steps_per_stage) * 100
                print(f"  [Step {self.current_step:3d}] Stage: {stage_name} ({stage_steps}/{self.il_steps_per_stage}, {progress_pct:.0f}%) | "
                      f"Quality: {stage_quality:.3f} | Stages: {stages_done}/3 | Reward: {reward:.2f}")
            
            if terminated or truncated:
                self._print_episode_summary(reward_info, terminated, truncated)
        
        info = {
            "step": self.current_step,
            "stage_info": self._il_wrapper.get_stage_info(),
            "stages_completed": self._stages_completed.copy(),
            "stage_advanced": stage_advanced,
            "is_success": terminated and all_stages_done,
            **reward_info,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_arm_only_residual(self, residual_action: np.ndarray) -> np.ndarray:
        """Get arm-only residual (hands are always zero)."""
        arm_residual = residual_action.copy()
        # Zero out hand joints - IL controls hands 100%
        arm_residual[6:30] = 0.0    # Left hand
        arm_residual[36:60] = 0.0   # Right hand
        return arm_residual
    
    def _execute_joint_action(self, joint_action: np.ndarray):
        """Execute joint-space action."""
        from isaacsim.core.utils.types import ArticulationAction
        
        action_left = ArticulationAction(joint_positions=joint_action[:30])
        action_right = ArticulationAction(joint_positions=joint_action[30:])
        
        self._robot.dexleft.apply_action(action_left)
        self._robot.dexright.apply_action(action_right)
        
        for _ in range(5):
            self._base_env.step()
    
    def _check_proactive_advance(self) -> bool:
        """Check if proactive advance should happen."""
        if not self.enable_proactive_advance:
            return False
        
        current_stage = self._il_wrapper.current_stage
        if current_stage == FoldingStage.COMPLETED:
            return False
        
        stage_steps = self._il_wrapper.stage_step_count
        stage_quality = self._evaluate_current_stage_quality()
        
        # Advance after duration threshold
        steps_threshold = int(self.il_steps_per_stage * self.proactive_advance_after_ratio)
        if stage_steps >= steps_threshold and stage_quality >= self.min_quality_for_proactive:
            return True
        
        # Or after full duration
        if stage_steps >= self.il_steps_per_stage:
            return True
        
        return False
    
    def _execute_proactive_advance(self) -> Tuple[bool, float]:
        """Execute proactive stage advance."""
        self._safe_release_before_transition()
        
        current_stage = self._il_wrapper.current_stage
        stage_idx = int(current_stage) - 1
        if 0 <= stage_idx < 3:
            self._stages_completed[stage_idx] = True
        
        success = self._il_wrapper.advance_stage()
        
        if success:
            self._stage_initial_bbox = self._compute_bbox(self._current_garment_pcd)
            self._update_garment_pcd_and_gam()
            self._pre_position_robot_for_stage()
            return True, self.reward_weights["stage_completion_bonus"]
        
        return False, 0.0
    
    def _handle_stage_advance(self) -> Tuple[bool, float]:
        """Handle RL-initiated stage advance."""
        current_stage = self._il_wrapper.current_stage
        stage_steps = self._il_wrapper.stage_step_count
        
        if current_stage == FoldingStage.COMPLETED:
            return False, 0.0
        
        if stage_steps < self.min_steps_before_advance:
            return False, -self.reward_weights["premature_advance_penalty"]
        
        stage_quality = self._evaluate_current_stage_quality()
        if stage_quality < 0.15:
            return False, -self.reward_weights["premature_advance_penalty"]
        
        self._safe_release_before_transition()
        
        stage_idx = int(current_stage) - 1
        self._stages_completed[stage_idx] = True
        
        success = self._il_wrapper.advance_stage()
        
        if success:
            self._stage_initial_bbox = self._compute_bbox(self._current_garment_pcd)
            self._update_garment_pcd_and_gam()
            self._pre_position_robot_for_stage()
            return True, self.reward_weights["stage_completion_bonus"]
        
        return False, 0.0
    
    def _safe_release_before_transition(self):
        """Safely release before stage transition."""
        self._robot.set_both_hand_state("open", "open")
        for _ in range(30):
            self._base_env.step()
        
        self._garment.particle_material.set_gravity_scale(10.0)
        for _ in range(50):
            self._base_env.step()
        self._garment.particle_material.set_gravity_scale(1.0)
        
        for _ in range(20):
            self._base_env.step()
    
    def _pre_position_robot_for_stage(self):
        """Pre-position robot for current stage."""
        if self._gam_keypoints is None:
            return
        
        try:
            current_stage = self._il_wrapper.current_stage
            
            if current_stage == FoldingStage.STAGE_1_LEFT_SLEEVE:
                target_pos = self._gam_keypoints[0].copy()
                target_pos[2] = 0.02
                self._robot.dexleft.dense_step_action(
                    target_pos=target_pos,
                    target_ori=np.array([0.579, -0.579, -0.406, 0.406]),
                    angular_type="quat"
                )
            elif current_stage == FoldingStage.STAGE_2_RIGHT_SLEEVE:
                target_pos = self._gam_keypoints[2].copy()
                target_pos[2] = 0.02
                self._robot.dexright.dense_step_action(
                    target_pos=target_pos,
                    target_ori=np.array([0.406, -0.406, -0.579, 0.579]),
                    angular_type="quat"
                )
            elif current_stage == FoldingStage.STAGE_3_BOTTOM_UP:
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
            
            for _ in range(20):
                self._base_env.step()
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️ Pre-positioning failed: {e}")
    
    def _update_garment_pcd_and_gam(self):
        """Update garment point cloud and GAM keypoints."""
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
        
        self._current_garment_pcd = self._normalize_pcd(pcd)
        
        # NaN protection
        if np.isnan(self._current_garment_pcd).any():
            self._current_garment_pcd = np.nan_to_num(self._current_garment_pcd, nan=0.0).astype(np.float32)
        
        # Get GAM keypoints
        try:
            keypoints, _, similarity = self._gam_model.get_manipulation_points(
                input_pcd=self._current_garment_pcd,
                index_list=[957, 501, 1902, 448, 1196, 422]
            )
            self._gam_keypoints = keypoints.astype(np.float32)
            
            # Update affordance
            current_stage = self._il_wrapper.current_stage
            config = STAGE_CONFIGS.get(current_stage) if hasattr(self._il_wrapper, 'current_stage') else None
            if config and similarity is not None:
                aff_indices = config.affordance_indices
                aff = np.stack([
                    similarity[aff_indices[0]],
                    similarity[aff_indices[1]]
                ], axis=-1)
                self._current_affordance = self._normalize_columns(aff).astype(np.float32)
            else:
                self._current_affordance = np.zeros((self.point_cloud_size, 2), dtype=np.float32)
        except:
            self._gam_keypoints = np.zeros((6, 3), dtype=np.float32)
            self._current_affordance = np.zeros((self.point_cloud_size, 2), dtype=np.float32)
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        left_joints = self._robot.dexleft.get_joint_positions()
        right_joints = self._robot.dexright.get_joint_positions()
        joint_positions = np.concatenate([left_joints, right_joints]).astype(np.float32)
        
        left_pos, left_ori = self._robot.dexleft.get_cur_ee_pos()
        right_pos, right_ori = self._robot.dexright.get_cur_ee_pos()
        ee_poses = np.concatenate([left_pos, left_ori, right_pos, right_ori]).astype(np.float32)
        
        if self._last_il_action is not None:
            il_action = self._last_il_action.astype(np.float32)
        else:
            il_obs = self._get_il_observation()
            il_action = self._get_il_joint_action(il_obs).astype(np.float32)
            self._last_il_action = il_action
        
        obs = {
            "garment_pcd": self._current_garment_pcd.astype(np.float32),
            "joint_positions": joint_positions,
            "ee_poses": ee_poses,
            "il_action": il_action,
            "gam_keypoints": self._gam_keypoints,
            "current_stage": self._il_wrapper.get_stage_one_hot(),
            "stage_progress": np.array([self._il_wrapper.get_stage_progress()], dtype=np.float32),
            "stages_completed": np.array(self._stages_completed, dtype=np.float32),
            "should_advance_hint": np.array([self._get_should_advance_hint()], dtype=np.float32),
        }
        
        # NaN protection
        for key, value in obs.items():
            if np.isnan(value).any() or np.isinf(value).any():
                obs[key] = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        return obs
    
    def _get_il_observation(self) -> Dict[str, np.ndarray]:
        """Get observation for IL policy."""
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
        """Get IL action in joint space."""
        joint_action = self._il_wrapper.get_single_step_action(il_obs)
        
        if len(joint_action) != self.joint_dim:
            if len(joint_action) < self.joint_dim:
                joint_action = np.pad(joint_action, (0, self.joint_dim - len(joint_action)))
            else:
                joint_action = joint_action[:self.joint_dim]
        
        # NaN protection
        if np.isnan(joint_action).any():
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
    
    def _evaluate_current_stage_quality(self) -> float:
        """Evaluate current stage fold quality."""
        if self._stage_initial_bbox is None:
            return 0.0
        
        current_bbox = self._compute_bbox(self._current_garment_pcd)
        current_size = current_bbox[3:6] - current_bbox[0:3]
        initial_size = self._stage_initial_bbox[3:6] - self._stage_initial_bbox[0:3]
        
        initial_area = initial_size[0] * initial_size[1]
        current_area = current_size[0] * current_size[1]
        
        if initial_area < 1e-6:
            return 0.0
        
        area_reduction = (initial_area - current_area) / initial_area
        return np.clip(area_reduction, 0.0, 1.0)
    
    def _get_should_advance_hint(self) -> float:
        """Get hint for RL about when to advance."""
        current_stage = self._il_wrapper.current_stage
        if current_stage == FoldingStage.COMPLETED:
            return 0.0
        
        stage_steps = self._il_wrapper.stage_step_count
        stage_quality = self._evaluate_current_stage_quality()
        
        step_ratio = stage_steps / self.il_steps_per_stage
        quality_ratio = stage_quality / 0.3
        
        combined = (step_ratio + quality_ratio) / 2.0
        
        if self._check_proactive_advance():
            combined = 1.0
        
        return np.clip(combined, 0.0, 1.0)
    
    def _compute_reward(self, obs: Dict[str, np.ndarray], residual: np.ndarray, stage_advanced: bool) -> Tuple[float, Dict]:
        """Compute SIMPLIFIED reward."""
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
        
        # Residual penalty
        residual_magnitude = np.sum(residual ** 2)
        reward -= residual_magnitude * self.reward_weights["residual_penalty"]
        info["residual_penalty"] = -residual_magnitude
        
        # Task completion
        if all(self._stages_completed) and self._check_final_success(obs):
            reward += self.reward_weights["task_success_bonus"]
            info["task_success_bonus"] = self.reward_weights["task_success_bonus"]
        
        info["total_reward"] = reward
        return reward, info
    
    def _check_final_success(self, obs: Dict[str, np.ndarray]) -> bool:
        """Check if task is complete."""
        pcd = obs["garment_pcd"]
        current_bbox = self._compute_bbox(pcd)
        current_size = current_bbox[3:6] - current_bbox[0:3]
        initial_size = self._initial_bbox[3:6] - self._initial_bbox[0:3]
        
        x_ratio = current_size[0] / (initial_size[0] + 1e-6)
        y_ratio = current_size[1] / (initial_size[1] + 1e-6)
        height_var = np.var(pcd[:, 2])
        
        return (x_ratio < 0.5) and (y_ratio < 0.7) and (height_var < 0.02)
    
    def _print_episode_summary(self, reward_info: Dict, terminated: bool, truncated: bool):
        """Print episode summary."""
        stages_done = sum(self._stages_completed)
        final_quality = reward_info.get("stage_quality", 0)
        total_reward = reward_info.get("total_reward", 0)
        
        current_stage = self._il_wrapper.current_stage
        stage_name = current_stage.name if hasattr(current_stage, 'name') else str(current_stage)
        stage_steps = self._il_wrapper.stage_step_count
        
        outcome = "✅ SUCCESS" if terminated else ("⏱️ TRUNCATED" if truncated else "❌ FAILED")
        
        print(f"\n{'='*60}")
        print(f"📊 Episode {self._episode_count} Summary (SIMPLIFIED)")
        print(f"{'='*60}")
        print(f"  Outcome:        {outcome}")
        print(f"  Steps:          {self.current_step}")
        print(f"  Current Stage:  {stage_name} (step {stage_steps})")
        print(f"  Stages Done:    {stages_done}/3")
        print(f"  Final Quality:  {final_quality:.3f}")
        print(f"  Total Reward:   {total_reward:.3f}")
        print(f"  RL Advances:    {self._total_rl_advances}")
        print(f"  Auto Advances:  {self._total_proactive_advances}")
        print(f"{'='*60}\n")
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._env_camera.get_rgb_graph(save_or_not=False)
        return None
    
    def close(self):
        """Clean up."""
        if self._initialized:
            self._base_env.stop()
            if self.verbose:
                print("[MultiStageResidualEnvSimple] Environment closed.")


def register_multi_stage_env_simple():
    """Register simplified environment."""
    gym.register(
        id="MultiStageResidualFoldTopsSimple-v0",
        entry_point="Env_RL.multi_stage_residual_env_simple:MultiStageResidualEnvSimple",
        max_episode_steps=400,
    )

