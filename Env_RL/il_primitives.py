"""
IL-Based Manipulation Primitives for Hierarchical RL.

This module replaces hard-coded IK primitives with trained SADP_G models.
Each primitive executes the corresponding stage's SADP_G model to completion.

Key Difference from primitives.py:
- OLD: Hand-coded IK trajectories for each folding action
- NEW: Trained diffusion policy (SADP_G) for each stage

This provides better performance since SADP_G was trained on demonstrations.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import IntEnum


class ILPrimitiveID(IntEnum):
    """
    Enumeration of IL-based manipulation primitives.
    
    Maps directly to SADP_G stages:
    - STAGE_1 (Left Sleeve): Uses stage_1 SADP_G model
    - STAGE_2 (Right Sleeve): Uses stage_2 SADP_G model
    - STAGE_3 (Bottom Fold): Uses stage_3 SADP_G model
    """
    STAGE_1_LEFT_SLEEVE = 0   # Execute SADP_G stage 1
    STAGE_2_RIGHT_SLEEVE = 1  # Execute SADP_G stage 2
    STAGE_3_BOTTOM_FOLD = 2   # Execute SADP_G stage 3
    OPEN_HANDS = 3            # Utility: Open grippers
    MOVE_TO_HOME = 4          # Utility: Move to home position
    DONE = 5                  # Signal completion


@dataclass
class ILPrimitiveResult:
    """Result of executing an IL primitive."""
    success: bool
    steps_taken: int
    info: Dict[str, Any]


class ILManipulationPrimitives:
    """
    IL-based manipulation primitives using trained SADP_G models.
    
    Instead of hand-coded IK trajectories, each folding primitive
    executes the corresponding SADP_G model's inference loop.
    
    Usage:
        primitives = ILManipulationPrimitives(
            robot, garment, base_env, garment_camera, env_camera,
            il_wrapper  # MultiStageSADPGWrapper
        )
        result = primitives.execute(ILPrimitiveID.STAGE_1_LEFT_SLEEVE, observation)
    """
    
    def __init__(
        self,
        robot,           # Bimanual_Ur10e instance
        garment,         # Particle_Garment instance
        base_env,        # BaseEnv instance
        garment_camera,  # Recording_Camera for garment point cloud
        env_camera,      # Recording_Camera for environment point cloud
        gam_model,       # GAM_Encapsulation for keypoints
        il_wrapper,      # MultiStageSADPGWrapper - THE KEY ADDITION!
        set_prim_visible_group_func=None,
        normalize_columns_func=None,
    ):
        """
        Initialize IL primitives with SADP_G wrapper.
        
        Args:
            robot: Bimanual UR10e robot instance
            garment: Particle garment instance
            base_env: Base environment for stepping physics
            garment_camera: Camera for getting garment point cloud
            env_camera: Camera for environment point cloud (depth)
            gam_model: GAM model for keypoint detection
            il_wrapper: MultiStageSADPGWrapper with loaded SADP_G models
            set_prim_visible_group_func: Function to hide/show prims
            normalize_columns_func: Function to normalize affordance features
        """
        self.robot = robot
        self.garment = garment
        self.base_env = base_env
        self.garment_camera = garment_camera
        self.env_camera = env_camera
        self.gam_model = gam_model
        self.il_wrapper = il_wrapper
        self._set_prim_visible_group = set_prim_visible_group_func
        self._normalize_columns = normalize_columns_func
        
        # Keypoint indices (same as SADP_G training)
        self.keypoint_indices = [957, 501, 1902, 448, 1196, 422]
        
        # Stage configs - max inference steps per stage
        self.stage_max_steps = {
            ILPrimitiveID.STAGE_1_LEFT_SLEEVE: 200,
            ILPrimitiveID.STAGE_2_RIGHT_SLEEVE: 200,
            ILPrimitiveID.STAGE_3_BOTTOM_FOLD: 200,
        }
        
        # Affordance indices for each stage (from SADP_G config)
        self.stage_affordance_indices = {
            ILPrimitiveID.STAGE_1_LEFT_SLEEVE: [0, 0],   # Left sleeve
            ILPrimitiveID.STAGE_2_RIGHT_SLEEVE: [2, 2],  # Right sleeve
            ILPrimitiveID.STAGE_3_BOTTOM_FOLD: [4, 5],   # Bottom corners
        }
        
        # Track which primitives have been executed
        self.executed_primitives = set()
        
        # Cached garment state
        self._current_garment_pcd = None
        self._current_affordance = None
        self._gam_similarity = None
        
    def reset(self):
        """Reset primitive execution state."""
        self.executed_primitives = set()
        self.il_wrapper.reset()
        self._current_garment_pcd = None
        self._current_affordance = None
        self._gam_similarity = None
        
    def execute(
        self, 
        primitive_id: int, 
        garment_pcd: np.ndarray,
    ) -> ILPrimitiveResult:
        """
        Execute an IL manipulation primitive.
        
        For folding primitives (STAGE_1, STAGE_2, STAGE_3), this runs
        the corresponding SADP_G model's inference loop until completion.
        
        Args:
            primitive_id: ID of the primitive to execute
            garment_pcd: Current garment point cloud
            
        Returns:
            ILPrimitiveResult with success status and info
        """
        primitive_id = ILPrimitiveID(primitive_id)
        
        # Execute the selected primitive
        if primitive_id == ILPrimitiveID.STAGE_1_LEFT_SLEEVE:
            return self._execute_il_stage(primitive_id, stage_num=1)
        elif primitive_id == ILPrimitiveID.STAGE_2_RIGHT_SLEEVE:
            return self._execute_il_stage(primitive_id, stage_num=2)
        elif primitive_id == ILPrimitiveID.STAGE_3_BOTTOM_FOLD:
            return self._execute_il_stage(primitive_id, stage_num=3)
        elif primitive_id == ILPrimitiveID.OPEN_HANDS:
            return self._open_hands()
        elif primitive_id == ILPrimitiveID.MOVE_TO_HOME:
            return self._move_to_home()
        elif primitive_id == ILPrimitiveID.DONE:
            return ILPrimitiveResult(success=True, steps_taken=0, info={"action": "done"})
        else:
            return ILPrimitiveResult(
                success=False,
                steps_taken=0,
                info={"error": f"Unknown primitive: {primitive_id}"}
            )
    
    def _execute_il_stage(
        self, 
        primitive_id: ILPrimitiveID, 
        stage_num: int
    ) -> ILPrimitiveResult:
        """
        Execute a SADP_G stage to completion.
        
        This is the core of IL-based primitives: instead of hand-coded
        IK trajectories, we run the trained diffusion policy.
        
        Args:
            primitive_id: The primitive being executed
            stage_num: Which SADP_G stage (1, 2, or 3)
            
        Returns:
            ILPrimitiveResult with execution details
        """
        from isaacsim.core.utils.types import ArticulationAction
        
        steps = 0
        max_steps = self.stage_max_steps[primitive_id]
        
        try:
            # Set IL wrapper to correct stage
            from Env_RL.multi_stage_sadpg_wrapper import FoldingStage
            stage_enum = FoldingStage(stage_num)
            self.il_wrapper.set_stage(stage_enum)
            
            # Update garment state and affordance for this stage
            self._update_garment_and_affordance(primitive_id)
            
            # ========== Pre-position Robot ==========
            # CRITICAL: The original SADP_G validation moves the robot to the
            # manipulation point BEFORE starting inference. Without this,
            # the robot is too far from the garment.
            self._pre_position_robot(primitive_id)
            steps += 20  # Pre-positioning steps
            
            # Execute IL policy for this stage
            for step in range(max_steps):
                # Get current observation for IL
                il_obs = self._get_il_observation(primitive_id)
                
                # Get action from SADP_G
                joint_action = self.il_wrapper.get_single_step_action(il_obs)
                
                # Execute action
                action_left = ArticulationAction(joint_positions=joint_action[:30])
                action_right = ArticulationAction(joint_positions=joint_action[30:])
                
                self.robot.dexleft.apply_action(action_left)
                self.robot.dexright.apply_action(action_right)
                
                # Step physics
                for _ in range(5):
                    self.base_env.step()
                
                steps += 5
                
                # Update IL observation history
                self.il_wrapper.update_obs(il_obs)
                
                # Update garment state periodically
                if step % 20 == 0:
                    self._update_garment_and_affordance(primitive_id)
                
                # Check for early completion (optional heuristic)
                # Could add fold quality check here
            
            # Let garment settle after stage completion
            self.garment.particle_material.set_gravity_scale(10.0)
            for _ in range(100):
                self.base_env.step()
            self.garment.particle_material.set_gravity_scale(1.0)
            steps += 100
            
            # Open hands after fold
            self.robot.set_both_hand_state("open", "open")
            for _ in range(30):
                self.base_env.step()
            steps += 30
            
            # Move arms away
            self._move_arms_away(primitive_id)
            steps += 50
            
            # Mark as executed
            self.executed_primitives.add(primitive_id)
            
            return ILPrimitiveResult(
                success=True,
                steps_taken=steps,
                info={
                    "primitive": primitive_id.name,
                    "stage": stage_num,
                    "il_steps": max_steps,
                }
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ILPrimitiveResult(
                success=False,
                steps_taken=steps,
                info={"error": str(e), "primitive": primitive_id.name}
            )
    
    def _get_il_observation(self, primitive_id: ILPrimitiveID) -> Dict[str, np.ndarray]:
        """Get observation in format expected by SADP_G."""
        # Get joint positions
        left_joints = self.robot.dexleft.get_joint_positions()
        right_joints = self.robot.dexright.get_joint_positions()
        joint_positions = np.concatenate([left_joints, right_joints])
        
        # Get environment point cloud
        env_pcd = self.env_camera.get_pointcloud_from_depth(
            show_original_pc_online=False,
            show_downsample_pc_online=False,
        )
        
        return {
            "agent_pos": joint_positions,
            "environment_point_cloud": env_pcd,
            "garment_point_cloud": self._current_garment_pcd,
            "points_affordance_feature": self._current_affordance,
        }
    
    def _update_garment_and_affordance(self, primitive_id: ILPrimitiveID):
        """Update cached garment point cloud and affordance features."""
        # Hide robots for clean point cloud
        if self._set_prim_visible_group:
            self._set_prim_visible_group(
                prim_path_list=["/World/DexLeft", "/World/DexRight"],
                visible=False,
            )
            for _ in range(2):
                self.base_env.step()
        
        # Get point cloud
        pcd, _ = self.garment_camera.get_point_cloud_data_from_segment(
            save_or_not=False,
            real_time_watch=False,
        )
        
        # Unhide robots
        if self._set_prim_visible_group:
            self._set_prim_visible_group(
                prim_path_list=["/World/DexLeft", "/World/DexRight"],
                visible=True,
            )
            for _ in range(2):
                self.base_env.step()
        
        self._current_garment_pcd = pcd
        
        # Get GAM keypoints and similarity
        try:
            keypoints, indices, similarity = self.gam_model.get_manipulation_points(
                input_pcd=pcd,
                index_list=self.keypoint_indices
            )
            self._gam_similarity = similarity
            
            # Build affordance for current stage
            aff_indices = self.stage_affordance_indices.get(primitive_id, [0, 0])
            aff = np.stack([
                similarity[aff_indices[0]],
                similarity[aff_indices[1]]
            ], axis=-1)
            
            if self._normalize_columns:
                self._current_affordance = self._normalize_columns(aff).astype(np.float32)
            else:
                self._current_affordance = aff.astype(np.float32)
                
        except Exception:
            # Fallback if GAM fails
            n_points = len(pcd) if pcd is not None else 2048
            self._current_affordance = np.zeros((n_points, 2), dtype=np.float32)
    
    def _pre_position_robot(self, primitive_id: ILPrimitiveID):
        """
        Pre-position the robot to the manipulation point before SADP_G inference.
        
        CRITICAL: The original SADP_G validation does this:
            env.bimanual_dex.dexleft.dense_step_action(target_pos=manipulation_points[0], ...)
            for i in range(20):
                env.step()
        
        Without this, the robot starts too far from the garment.
        """
        if self._gam_similarity is None:
            return
        
        try:
            # Get manipulation points from GAM
            keypoints, _, _ = self.gam_model.get_manipulation_points(
                input_pcd=self._current_garment_pcd,
                index_list=self.keypoint_indices
            )
            
            if primitive_id == ILPrimitiveID.STAGE_1_LEFT_SLEEVE:
                # Move left hand to left sleeve tip (keypoint 0)
                target_pos = keypoints[0].copy()
                target_pos[2] = 0.02  # Slightly above ground
                target_ori = np.array([0.579, -0.579, -0.406, 0.406])
                
                self.robot.dexleft.dense_step_action(
                    target_pos=target_pos,
                    target_ori=target_ori,
                    angular_type="quat"
                )
                
            elif primitive_id == ILPrimitiveID.STAGE_2_RIGHT_SLEEVE:
                # Move right hand to right sleeve tip (keypoint 2)
                target_pos = keypoints[2].copy()
                target_pos[2] = 0.02
                target_ori = np.array([0.406, -0.406, -0.579, 0.579])
                
                self.robot.dexright.dense_step_action(
                    target_pos=target_pos,
                    target_ori=target_ori,
                    angular_type="quat"
                )
                
            elif primitive_id == ILPrimitiveID.STAGE_3_BOTTOM_FOLD:
                # Move both hands to bottom corners (keypoints 4 and 5)
                left_pos = keypoints[4].copy()
                right_pos = keypoints[5].copy()
                left_pos[2] = 0.0
                right_pos[2] = 0.0
                
                self.robot.dense_move_both_ik(
                    left_pos=left_pos,
                    left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
                    right_pos=right_pos,
                    right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
                )
            
            # Settle after pre-positioning
            for _ in range(20):
                self.base_env.step()
                
        except Exception as e:
            print(f"[WARNING] Pre-positioning failed: {e}")
    
    def _move_arms_away(self, primitive_id: ILPrimitiveID):
        """Move arms away after completing a fold."""
        try:
            if primitive_id == ILPrimitiveID.STAGE_1_LEFT_SLEEVE:
                # Move left arm away
                self.robot.dexleft.dense_step_action(
                    target_pos=np.array([-0.6, 0.8, 0.5]),
                    target_ori=np.array([0.579, -0.579, -0.406, 0.406]),
                    angular_type="quat"
                )
            elif primitive_id == ILPrimitiveID.STAGE_2_RIGHT_SLEEVE:
                # Move right arm away
                self.robot.dexright.dense_step_action(
                    target_pos=np.array([0.6, 0.8, 0.5]),
                    target_ori=np.array([0.406, -0.406, -0.579, 0.579]),
                    angular_type="quat"
                )
            else:
                # Move both arms away for bottom fold
                self.robot.dexleft.dense_step_action(
                    target_pos=np.array([-0.6, 0.8, 0.5]),
                    target_ori=np.array([0.579, -0.579, -0.406, 0.406]),
                    angular_type="quat"
                )
                self.robot.dexright.dense_step_action(
                    target_pos=np.array([0.6, 0.8, 0.5]),
                    target_ori=np.array([0.406, -0.406, -0.579, 0.579]),
                    angular_type="quat"
                )
        except Exception:
            pass
    
    def _open_hands(self) -> ILPrimitiveResult:
        """Open both robot hands."""
        self.robot.set_both_hand_state("open", "open")
        for _ in range(30):
            self.base_env.step()
        return ILPrimitiveResult(
            success=True,
            steps_taken=30,
            info={"primitive": "open_hands"}
        )
    
    def _move_to_home(self) -> ILPrimitiveResult:
        """Move both arms to home position."""
        steps = 0
        
        try:
            self.robot.dexleft.dense_step_action(
                target_pos=np.array([-0.6, 0.5, 0.5]),
                target_ori=np.array([0.579, -0.579, -0.406, 0.406]),
                angular_type="quat"
            )
            steps += 50
            
            self.robot.dexright.dense_step_action(
                target_pos=np.array([0.6, 0.5, 0.5]),
                target_ori=np.array([0.406, -0.406, -0.579, 0.579]),
                angular_type="quat"
            )
            steps += 50
            
            return ILPrimitiveResult(
                success=True,
                steps_taken=steps,
                info={"primitive": "move_to_home"}
            )
            
        except Exception as e:
            return ILPrimitiveResult(
                success=False,
                steps_taken=steps,
                info={"error": str(e), "primitive": "move_to_home"}
            )
    
    def get_executed_primitives(self) -> set:
        """Get set of primitives that have been executed."""
        return self.executed_primitives.copy()
    
    def is_complete_sequence(self) -> bool:
        """Check if all required folding stages have been executed."""
        required = {
            ILPrimitiveID.STAGE_1_LEFT_SLEEVE,
            ILPrimitiveID.STAGE_2_RIGHT_SLEEVE,
            ILPrimitiveID.STAGE_3_BOTTOM_FOLD
        }
        return required.issubset(self.executed_primitives)
