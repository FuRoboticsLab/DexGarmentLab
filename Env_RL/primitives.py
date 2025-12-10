"""
Manipulation Primitives for Hierarchical RL.

This module defines low-level manipulation primitives that wrap existing
DexGarmentLab functionality. The high-level RL policy selects which
primitive to execute, and these functions handle the actual robot control.

Each primitive is a self-contained action sequence using the existing
robot control methods (IK, grasp, etc.) from DexGarmentLab.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import IntEnum


class PrimitiveID(IntEnum):
    """Enumeration of available manipulation primitives."""
    FOLD_LEFT_SLEEVE = 0
    FOLD_RIGHT_SLEEVE = 1
    FOLD_BOTTOM = 2
    OPEN_HANDS = 3
    MOVE_TO_HOME = 4
    DONE = 5


@dataclass
class PrimitiveResult:
    """Result of executing a primitive."""
    success: bool
    steps_taken: int
    info: Dict[str, Any]


class ManipulationPrimitives:
    """
    Collection of manipulation primitives for garment folding.
    
    Each primitive wraps existing DexGarmentLab robot control methods
    to perform a complete manipulation action (approach → grasp → move → release).
    
    Usage:
        primitives = ManipulationPrimitives(robot, garment, gam_model, env)
        result = primitives.execute(PrimitiveID.FOLD_LEFT_SLEEVE, observation)
    """
    
    def __init__(
        self,
        robot,           # Bimanual_Ur10e instance
        garment,         # Particle_Garment instance
        gam_model,       # GAM_Encapsulation instance
        base_env,        # BaseEnv instance
        garment_camera,  # Recording_Camera for point cloud
    ):
        """
        Initialize primitives with robot and environment references.
        
        Args:
            robot: Bimanual UR10e robot instance
            garment: Particle garment instance
            gam_model: GAM model for keypoint detection
            base_env: Base environment for stepping physics
            garment_camera: Camera for getting garment point cloud
        """
        self.robot = robot
        self.garment = garment
        self.gam_model = gam_model
        self.base_env = base_env
        self.garment_camera = garment_camera
        
        # Keypoint indices for different parts of the garment
        # These are the same indices used in Fold_Tops_Env.py
        self.keypoint_indices = [957, 501, 1902, 448, 1196, 422]
        # Index mapping:
        # 0: left sleeve tip, 1: left sleeve target
        # 2: right sleeve tip, 3: right sleeve target
        # 4: bottom left, 5: bottom right
        
        # Track which primitives have been executed
        self.executed_primitives = set()
        
    def reset(self):
        """Reset primitive execution state."""
        self.executed_primitives = set()
        
    def execute(
        self, 
        primitive_id: int, 
        garment_pcd: np.ndarray,
        set_prim_visible_group_func=None,
    ) -> PrimitiveResult:
        """
        Execute a manipulation primitive.
        
        Args:
            primitive_id: ID of the primitive to execute
            garment_pcd: Current garment point cloud
            set_prim_visible_group_func: Function to hide/show robot for point cloud
            
        Returns:
            PrimitiveResult with success status and info
        """
        primitive_id = PrimitiveID(primitive_id)
        
        # Get manipulation points from GAM
        try:
            manipulation_points, indices, points_similarity = self.gam_model.get_manipulation_points(
                input_pcd=garment_pcd,
                index_list=self.keypoint_indices
            )
            # Set Z coordinates for grasping
            manipulation_points[0:4, 2] = 0.02  # Sleeve points slightly above ground
            manipulation_points[4:, 2] = 0.0    # Bottom points at ground level
        except Exception as e:
            return PrimitiveResult(
                success=False,
                steps_taken=0,
                info={"error": f"GAM failed: {str(e)}"}
            )
        
        # Execute the selected primitive
        if primitive_id == PrimitiveID.FOLD_LEFT_SLEEVE:
            return self._fold_left_sleeve(manipulation_points)
        elif primitive_id == PrimitiveID.FOLD_RIGHT_SLEEVE:
            return self._fold_right_sleeve(manipulation_points)
        elif primitive_id == PrimitiveID.FOLD_BOTTOM:
            return self._fold_bottom(manipulation_points)
        elif primitive_id == PrimitiveID.OPEN_HANDS:
            return self._open_hands()
        elif primitive_id == PrimitiveID.MOVE_TO_HOME:
            return self._move_to_home()
        elif primitive_id == PrimitiveID.DONE:
            return PrimitiveResult(success=True, steps_taken=0, info={"action": "done"})
        else:
            return PrimitiveResult(
                success=False,
                steps_taken=0,
                info={"error": f"Unknown primitive: {primitive_id}"}
            )
    
    def _fold_left_sleeve(self, manipulation_points: np.ndarray) -> PrimitiveResult:
        """
        Execute left sleeve folding primitive.
        
        Sequence:
        1. Move left hand to sleeve tip (manipulation_points[0])
        2. Close gripper
        3. Lift sleeve
        4. Move to fold target (manipulation_points[1])
        5. Release
        """
        steps = 0
        
        try:
            # 1. Approach left sleeve tip
            self.robot.dexleft.dense_step_action(
                target_pos=manipulation_points[0],
                target_ori=np.array([0.579, -0.579, -0.406, 0.406]),
                angular_type="quat"
            )
            steps += 50
            
            # 2. Close left gripper
            self.robot.set_both_hand_state(left_hand_state="close", right_hand_state="None")
            for _ in range(30):
                self.base_env.step()
            steps += 30
            
            # 3. Compute lift height
            left_sleeve_height = min(
                np.linalg.norm(manipulation_points[0][:2] - manipulation_points[3][:2]),
                0.3
            )
            
            # 4. Lift sleeve
            lift_point_1 = np.array([
                manipulation_points[0][0],
                manipulation_points[0][1],
                left_sleeve_height
            ])
            self.robot.dexleft.dense_step_action(
                target_pos=lift_point_1,
                target_ori=np.array([0.579, -0.579, -0.406, 0.406]),
                angular_type="quat"
            )
            steps += 30
            
            # 5. Move to fold target
            lift_point_2 = np.array([
                manipulation_points[1][0],
                manipulation_points[1][1],
                left_sleeve_height
            ])
            self.robot.dexleft.dense_step_action(
                target_pos=lift_point_2,
                target_ori=np.array([0.579, -0.579, -0.406, 0.406]),
                angular_type="quat"
            )
            steps += 30
            
            # 6. Release
            self.robot.set_both_hand_state(left_hand_state="open", right_hand_state="None")
            for _ in range(30):
                self.base_env.step()
            steps += 30
            
            # 7. Let garment settle with increased gravity
            self.garment.particle_material.set_gravity_scale(10.0)
            for _ in range(100):
                self.base_env.step()
            self.garment.particle_material.set_gravity_scale(1.0)
            steps += 100
            
            # 8. Move arm away
            self.robot.dexleft.dense_step_action(
                target_pos=np.array([-0.6, 0.8, 0.5]),
                target_ori=np.array([0.579, -0.579, -0.406, 0.406]),
                angular_type="quat"
            )
            steps += 30
            
            self.executed_primitives.add(PrimitiveID.FOLD_LEFT_SLEEVE)
            
            return PrimitiveResult(
                success=True,
                steps_taken=steps,
                info={"primitive": "fold_left_sleeve"}
            )
            
        except Exception as e:
            return PrimitiveResult(
                success=False,
                steps_taken=steps,
                info={"error": str(e), "primitive": "fold_left_sleeve"}
            )
    
    def _fold_right_sleeve(self, manipulation_points: np.ndarray) -> PrimitiveResult:
        """
        Execute right sleeve folding primitive.
        
        Mirror of left sleeve folding using right arm.
        """
        steps = 0
        
        try:
            # 1. Approach right sleeve tip
            self.robot.dexright.dense_step_action(
                target_pos=manipulation_points[2],
                target_ori=np.array([0.406, -0.406, -0.579, 0.579]),
                angular_type="quat"
            )
            steps += 50
            
            # 2. Close right gripper
            self.robot.set_both_hand_state(left_hand_state="None", right_hand_state="close")
            for _ in range(30):
                self.base_env.step()
            steps += 30
            
            # 3. Compute lift height
            right_sleeve_height = min(
                np.linalg.norm(manipulation_points[2][:2] - manipulation_points[1][:2]),
                0.3
            )
            
            # 4. Lift sleeve
            lift_point_1 = np.array([
                manipulation_points[2][0],
                manipulation_points[2][1],
                right_sleeve_height
            ])
            self.robot.dexright.dense_step_action(
                target_pos=lift_point_1,
                target_ori=np.array([0.406, -0.406, -0.579, 0.579]),
                angular_type="quat"
            )
            steps += 30
            
            # 5. Move to fold target
            lift_point_2 = np.array([
                manipulation_points[3][0],
                manipulation_points[3][1],
                right_sleeve_height
            ])
            self.robot.dexright.dense_step_action(
                target_pos=lift_point_2,
                target_ori=np.array([0.406, -0.406, -0.579, 0.579]),
                angular_type="quat"
            )
            steps += 30
            
            # 6. Release
            self.robot.set_both_hand_state(left_hand_state="None", right_hand_state="open")
            for _ in range(30):
                self.base_env.step()
            steps += 30
            
            # 7. Let garment settle
            self.garment.particle_material.set_gravity_scale(10.0)
            for _ in range(100):
                self.base_env.step()
            self.garment.particle_material.set_gravity_scale(1.0)
            steps += 100
            
            # 8. Move arm away
            self.robot.dexright.dense_step_action(
                target_pos=np.array([0.6, 0.8, 0.5]),
                target_ori=np.array([0.406, -0.406, -0.579, 0.579]),
                angular_type="quat"
            )
            steps += 30
            
            self.executed_primitives.add(PrimitiveID.FOLD_RIGHT_SLEEVE)
            
            return PrimitiveResult(
                success=True,
                steps_taken=steps,
                info={"primitive": "fold_right_sleeve"}
            )
            
        except Exception as e:
            return PrimitiveResult(
                success=False,
                steps_taken=steps,
                info={"error": str(e), "primitive": "fold_right_sleeve"}
            )
    
    def _fold_bottom(self, manipulation_points: np.ndarray) -> PrimitiveResult:
        """
        Execute bottom-to-top folding primitive.
        
        Uses both arms to grasp bottom corners and fold upward.
        """
        steps = 0
        
        try:
            # 1. Move both hands to bottom corners
            self.robot.dense_move_both_ik(
                left_pos=manipulation_points[4],
                left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
                right_pos=manipulation_points[5],
                right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
            )
            steps += 50
            
            # 2. Close both grippers
            self.robot.set_both_hand_state(left_hand_state="close", right_hand_state="close")
            for _ in range(30):
                self.base_env.step()
            steps += 30
            
            # 3. Compute lift height
            lift_height = manipulation_points[3][1] - manipulation_points[4][1]
            
            # 4. Lift both corners
            lift_point_1 = np.array([
                manipulation_points[4][0],
                manipulation_points[4][1],
                lift_height / 2
            ])
            lift_point_2 = np.array([
                manipulation_points[5][0],
                manipulation_points[5][1],
                lift_height / 2
            ])
            self.robot.dense_move_both_ik(
                left_pos=lift_point_1,
                left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
                right_pos=lift_point_2,
                right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
            )
            steps += 50
            
            # 5. Push forward to complete fold
            push_point_1 = np.array([
                manipulation_points[3][0],
                manipulation_points[3][1] + 0.1,
                min(lift_height / 2, 0.2)
            ])
            push_point_2 = np.array([
                manipulation_points[1][0],
                manipulation_points[1][1] + 0.1,
                min(lift_height / 2, 0.2)
            ])
            self.robot.dense_move_both_ik(
                left_pos=push_point_1,
                left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
                right_pos=push_point_2,
                right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
            )
            steps += 50
            
            # 6. Release both
            self.robot.set_both_hand_state(left_hand_state="open", right_hand_state="open")
            for _ in range(30):
                self.base_env.step()
            steps += 30
            
            # 7. Let garment settle
            self.garment.particle_material.set_gravity_scale(10.0)
            for _ in range(100):
                self.base_env.step()
            self.garment.particle_material.set_gravity_scale(1.0)
            steps += 100
            
            self.executed_primitives.add(PrimitiveID.FOLD_BOTTOM)
            
            return PrimitiveResult(
                success=True,
                steps_taken=steps,
                info={"primitive": "fold_bottom"}
            )
            
        except Exception as e:
            return PrimitiveResult(
                success=False,
                steps_taken=steps,
                info={"error": str(e), "primitive": "fold_bottom"}
            )
    
    def _open_hands(self) -> PrimitiveResult:
        """Open both robot hands."""
        self.robot.set_both_hand_state("open", "open")
        for _ in range(30):
            self.base_env.step()
        return PrimitiveResult(
            success=True,
            steps_taken=30,
            info={"primitive": "open_hands"}
        )
    
    def _move_to_home(self) -> PrimitiveResult:
        """Move both arms to home position."""
        steps = 0
        
        try:
            # Move left arm to home
            self.robot.dexleft.dense_step_action(
                target_pos=np.array([-0.6, 0.5, 0.5]),
                target_ori=np.array([0.579, -0.579, -0.406, 0.406]),
                angular_type="quat"
            )
            steps += 50
            
            # Move right arm to home
            self.robot.dexright.dense_step_action(
                target_pos=np.array([0.6, 0.5, 0.5]),
                target_ori=np.array([0.406, -0.406, -0.579, 0.579]),
                angular_type="quat"
            )
            steps += 50
            
            return PrimitiveResult(
                success=True,
                steps_taken=steps,
                info={"primitive": "move_to_home"}
            )
            
        except Exception as e:
            return PrimitiveResult(
                success=False,
                steps_taken=steps,
                info={"error": str(e), "primitive": "move_to_home"}
            )
    
    def get_executed_primitives(self) -> set:
        """Get set of primitives that have been executed."""
        return self.executed_primitives.copy()
    
    def is_complete_sequence(self) -> bool:
        """Check if all required folding primitives have been executed."""
        required = {
            PrimitiveID.FOLD_LEFT_SLEEVE,
            PrimitiveID.FOLD_RIGHT_SLEEVE,
            PrimitiveID.FOLD_BOTTOM
        }
        return required.issubset(self.executed_primitives)



