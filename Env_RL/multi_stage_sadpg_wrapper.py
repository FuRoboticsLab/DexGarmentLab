"""
Multi-Stage SADP_G Wrapper for Residual RL.

This module provides a unified interface to manage all 3 SADP_G stage models
and provides stage-appropriate IL guidance for residual PPO training.

The wrapper handles:
- Loading all 3 stage checkpoints
- Tracking current stage
- Providing IL actions from the appropriate stage model
- Managing observation history for each stage's diffusion policy

Usage:
    wrapper = MultiStageSADPGWrapper(
        training_data_num=100,
        stage_1_checkpoint=1500,
        stage_2_checkpoint=1500,
        stage_3_checkpoint=1500,
    )
    
    # Get IL action for current stage
    il_action = wrapper.get_action(observation, stage=1)
    
    # Or let wrapper track stage internally
    wrapper.set_stage(2)
    il_action = wrapper.get_action(observation)
"""

import os
import sys
import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import IntEnum

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FoldingStage(IntEnum):
    """Folding stages for the tops garment task."""
    STAGE_1_LEFT_SLEEVE = 1   # Fold left sleeve inward
    STAGE_2_RIGHT_SLEEVE = 2  # Fold right sleeve inward
    STAGE_3_BOTTOM_UP = 3     # Fold bottom half up
    COMPLETED = 4             # All folds done


@dataclass
class StageConfig:
    """Configuration for each stage."""
    stage_id: int
    task_name: str
    max_inference_steps: int  # Max steps for this stage
    description: str
    # GAM keypoint indices used for this stage
    keypoint_indices: List[int]
    # Which similarity indices to use for affordance
    affordance_indices: List[int]


# Default stage configurations matching Fold_Tops_HALO.py
STAGE_CONFIGS = {
    FoldingStage.STAGE_1_LEFT_SLEEVE: StageConfig(
        stage_id=1,
        task_name="Fold_Tops_stage_1",
        max_inference_steps=8,
        description="Fold left sleeve inward",
        keypoint_indices=[957, 501, 1902, 448, 1196, 422],
        affordance_indices=[0, 0],  # Uses similarity[0:1] twice
    ),
    FoldingStage.STAGE_2_RIGHT_SLEEVE: StageConfig(
        stage_id=2,
        task_name="Fold_Tops_stage_2",
        max_inference_steps=8,
        description="Fold right sleeve inward",
        keypoint_indices=[957, 501, 1902, 448, 1196, 422],
        affordance_indices=[2, 2],  # Uses similarity[2:3] twice
    ),
    FoldingStage.STAGE_3_BOTTOM_UP: StageConfig(
        stage_id=3,
        task_name="Fold_Tops_stage_3",
        max_inference_steps=12,
        description="Fold bottom half up",
        keypoint_indices=[957, 501, 1902, 448, 1196, 422],
        affordance_indices=[4, 5],  # Uses similarity[4:6]
    ),
}


class MultiStageSADPGWrapper:
    """
    Unified wrapper for all 3 SADP_G stage models.
    
    Provides a single interface to get IL actions from the appropriate
    stage model based on current folding progress.
    """
    
    def __init__(
        self,
        training_data_num: int = 100,
        stage_1_checkpoint: int = 1500,
        stage_2_checkpoint: int = 1500,
        stage_3_checkpoint: int = 1500,
        device: str = "cuda:0",
        lazy_load: bool = True,
    ):
        """
        Initialize multi-stage SADP_G wrapper.
        
        Args:
            training_data_num: Number of training data used for checkpoints
            stage_1_checkpoint: Checkpoint number for stage 1
            stage_2_checkpoint: Checkpoint number for stage 2
            stage_3_checkpoint: Checkpoint number for stage 3
            device: Device for inference
            lazy_load: If True, load models only when first needed
        """
        self.training_data_num = training_data_num
        self.checkpoints = {
            FoldingStage.STAGE_1_LEFT_SLEEVE: stage_1_checkpoint,
            FoldingStage.STAGE_2_RIGHT_SLEEVE: stage_2_checkpoint,
            FoldingStage.STAGE_3_BOTTOM_UP: stage_3_checkpoint,
        }
        self.device = device
        self.lazy_load = lazy_load
        
        # Stage models (loaded lazily or immediately)
        self._models: Dict[FoldingStage, any] = {}
        self._loaded = False
        
        # Current stage tracking
        self._current_stage = FoldingStage.STAGE_1_LEFT_SLEEVE
        self._stage_step_count = 0  # Steps within current stage
        
        # Action dimensions
        self.joint_action_dim = 60  # 30 per arm+hand
        self.ee_action_dim = 8      # EE-space: xyz*2 + grip*2
        
        # N action steps from diffusion policy
        self.n_action_steps = 4
        
        if not lazy_load:
            self.load_all_models()
    
    def load_all_models(self):
        """Load all 3 SADP_G models."""
        if self._loaded:
            return
        
        print("[MultiStageSADPGWrapper] Loading all SADP_G models...")
        
        try:
            from Model_HALO.SADP_G.SADP_G import SADP_G
            
            for stage in [FoldingStage.STAGE_1_LEFT_SLEEVE, 
                         FoldingStage.STAGE_2_RIGHT_SLEEVE,
                         FoldingStage.STAGE_3_BOTTOM_UP]:
                config = STAGE_CONFIGS[stage]
                checkpoint = self.checkpoints[stage]
                
                print(f"  Loading {config.task_name} (checkpoint {checkpoint})...")
                
                self._models[stage] = SADP_G(
                    task_name=config.task_name,
                    checkpoint_num=checkpoint,
                    data_num=self.training_data_num,
                    device=self.device,
                )
            
            self._loaded = True
            print("[MultiStageSADPGWrapper] All models loaded successfully!")
            
        except Exception as e:
            print(f"[MultiStageSADPGWrapper] Failed to load models: {e}")
            raise
    
    def _ensure_stage_loaded(self, stage: FoldingStage):
        """Ensure the model for a specific stage is loaded."""
        if stage not in self._models:
            if not self._loaded:
                self.load_all_models()
            if stage not in self._models:
                raise RuntimeError(f"Failed to load model for {stage}")
    
    @property
    def current_stage(self) -> FoldingStage:
        """Get current folding stage."""
        return self._current_stage
    
    @property
    def stage_step_count(self) -> int:
        """Get number of steps taken in current stage."""
        return self._stage_step_count
    
    @property
    def current_stage_config(self) -> StageConfig:
        """Get configuration for current stage."""
        return STAGE_CONFIGS.get(self._current_stage)
    
    def set_stage(self, stage: int):
        """
        Set the current stage.
        
        Args:
            stage: Stage number (1, 2, or 3)
        """
        if stage == 1:
            self._current_stage = FoldingStage.STAGE_1_LEFT_SLEEVE
        elif stage == 2:
            self._current_stage = FoldingStage.STAGE_2_RIGHT_SLEEVE
        elif stage == 3:
            self._current_stage = FoldingStage.STAGE_3_BOTTOM_UP
        elif stage == 4:
            self._current_stage = FoldingStage.COMPLETED
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 1, 2, 3, or 4.")
        
        self._stage_step_count = 0
        
        # Reset observation history for the new stage's model
        if self._current_stage != FoldingStage.COMPLETED and self._loaded:
            self._models[self._current_stage].env_runner.reset_obs()
    
    def advance_stage(self) -> bool:
        """
        Advance to the next stage.
        
        Returns:
            True if successfully advanced, False if already completed
        """
        if self._current_stage == FoldingStage.STAGE_1_LEFT_SLEEVE:
            self.set_stage(2)
            return True
        elif self._current_stage == FoldingStage.STAGE_2_RIGHT_SLEEVE:
            self.set_stage(3)
            return True
        elif self._current_stage == FoldingStage.STAGE_3_BOTTOM_UP:
            self._current_stage = FoldingStage.COMPLETED
            return True
        return False
    
    def reset(self):
        """Reset to initial state (stage 1)."""
        self._current_stage = FoldingStage.STAGE_1_LEFT_SLEEVE
        self._stage_step_count = 0
        
        # Reset observation history for all models
        if self._loaded:
            for model in self._models.values():
                model.env_runner.reset_obs()
    
    def get_action(
        self,
        observation: Dict[str, np.ndarray],
        stage: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get action from the appropriate SADP_G model.
        
        Args:
            observation: Dictionary containing:
                - agent_pos: (60,) joint positions
                - environment_point_cloud: (N, 3) environment point cloud
                - garment_point_cloud: (N, 3) garment point cloud
                - points_affordance_feature: (N, 2) affordance features
            stage: Optional stage override (1, 2, or 3). If None, uses current stage.
            
        Returns:
            action: (n_action_steps, 60) joint action sequence or (60,) single action
        """
        # Determine which stage model to use
        if stage is not None:
            if stage == 1:
                target_stage = FoldingStage.STAGE_1_LEFT_SLEEVE
            elif stage == 2:
                target_stage = FoldingStage.STAGE_2_RIGHT_SLEEVE
            elif stage == 3:
                target_stage = FoldingStage.STAGE_3_BOTTOM_UP
            else:
                raise ValueError(f"Invalid stage: {stage}")
        else:
            target_stage = self._current_stage
        
        # Check if completed
        if target_stage == FoldingStage.COMPLETED:
            return np.zeros((self.n_action_steps, self.joint_action_dim), dtype=np.float32)
        
        # Ensure model is loaded
        self._ensure_stage_loaded(target_stage)
        
        # Get action from model
        model = self._models[target_stage]
        action = model.get_action(observation)
        
        # Increment step count
        self._stage_step_count += 1
        
        return action
    
    def get_single_step_action(
        self,
        observation: Dict[str, np.ndarray],
        stage: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get single-step action (first action in sequence).
        
        Args:
            observation: Observation dict
            stage: Optional stage override
            
        Returns:
            action: (60,) single joint action
        """
        action = self.get_action(observation, stage)
        
        if len(action.shape) > 1:
            return action[0]
        return action
    
    def update_obs(self, observation: Dict[str, np.ndarray], stage: Optional[int] = None):
        """
        Update observation history for the appropriate stage model.
        
        This is important for diffusion policies that use temporal observations.
        
        Args:
            observation: Current observation
            stage: Optional stage override
        """
        if stage is not None:
            if stage == 1:
                target_stage = FoldingStage.STAGE_1_LEFT_SLEEVE
            elif stage == 2:
                target_stage = FoldingStage.STAGE_2_RIGHT_SLEEVE
            elif stage == 3:
                target_stage = FoldingStage.STAGE_3_BOTTOM_UP
            else:
                return
        else:
            target_stage = self._current_stage
        
        if target_stage == FoldingStage.COMPLETED:
            return
        
        if target_stage in self._models:
            self._models[target_stage].update_obs(observation)
    
    def get_stage_info(self) -> Dict:
        """Get information about current stage."""
        config = self.current_stage_config
        
        return {
            "stage": int(self._current_stage),
            "stage_name": self._current_stage.name,
            "step_count": self._stage_step_count,
            "max_steps": config.max_inference_steps if config else 0,
            "description": config.description if config else "Completed",
            "is_completed": self._current_stage == FoldingStage.COMPLETED,
        }
    
    def should_consider_stage_advance(self) -> bool:
        """
        Check if we should consider advancing to next stage.
        
        Based on step count reaching a reasonable threshold.
        The actual decision should be made by RL policy.
        """
        config = self.current_stage_config
        if config is None:
            return False
        
        # Consider advancing after at least half the max steps
        return self._stage_step_count >= config.max_inference_steps // 2
    
    def get_stage_one_hot(self) -> np.ndarray:
        """
        Get one-hot encoding of current stage.
        
        Returns:
            one_hot: (4,) array - [stage1, stage2, stage3, completed]
        """
        one_hot = np.zeros(4, dtype=np.float32)
        one_hot[int(self._current_stage) - 1] = 1.0
        return one_hot
    
    def get_stage_progress(self) -> float:
        """
        Get normalized progress within current stage.
        
        Returns:
            progress: float in [0, 1]
        """
        config = self.current_stage_config
        if config is None:
            return 1.0
        
        return min(1.0, self._stage_step_count / config.max_inference_steps)


class DummyMultiStageSADPG:
    """
    Dummy multi-stage wrapper for testing without loading actual models.
    
    Returns zero actions, useful for debugging the environment setup.
    """
    
    def __init__(self, **kwargs):
        self._current_stage = FoldingStage.STAGE_1_LEFT_SLEEVE
        self._stage_step_count = 0
        self.joint_action_dim = 60
        self.ee_action_dim = 8
        self.n_action_steps = 4
        self._loaded = True
    
    @property
    def current_stage(self) -> FoldingStage:
        return self._current_stage
    
    @property
    def stage_step_count(self) -> int:
        return self._stage_step_count
    
    @property
    def current_stage_config(self) -> StageConfig:
        return STAGE_CONFIGS.get(self._current_stage)
    
    def load_all_models(self):
        pass
    
    def set_stage(self, stage: int):
        if stage == 1:
            self._current_stage = FoldingStage.STAGE_1_LEFT_SLEEVE
        elif stage == 2:
            self._current_stage = FoldingStage.STAGE_2_RIGHT_SLEEVE
        elif stage == 3:
            self._current_stage = FoldingStage.STAGE_3_BOTTOM_UP
        elif stage == 4:
            self._current_stage = FoldingStage.COMPLETED
        self._stage_step_count = 0
    
    def advance_stage(self) -> bool:
        if self._current_stage == FoldingStage.STAGE_1_LEFT_SLEEVE:
            self.set_stage(2)
            return True
        elif self._current_stage == FoldingStage.STAGE_2_RIGHT_SLEEVE:
            self.set_stage(3)
            return True
        elif self._current_stage == FoldingStage.STAGE_3_BOTTOM_UP:
            self._current_stage = FoldingStage.COMPLETED
            return True
        return False
    
    def reset(self):
        self._current_stage = FoldingStage.STAGE_1_LEFT_SLEEVE
        self._stage_step_count = 0
    
    def get_action(self, observation: Dict, stage: Optional[int] = None) -> np.ndarray:
        self._stage_step_count += 1
        return np.zeros((self.n_action_steps, self.joint_action_dim), dtype=np.float32)
    
    def get_single_step_action(self, observation: Dict, stage: Optional[int] = None) -> np.ndarray:
        self._stage_step_count += 1
        return np.zeros(self.joint_action_dim, dtype=np.float32)
    
    def update_obs(self, observation: Dict, stage: Optional[int] = None):
        pass
    
    def get_stage_info(self) -> Dict:
        config = self.current_stage_config
        return {
            "stage": int(self._current_stage),
            "stage_name": self._current_stage.name,
            "step_count": self._stage_step_count,
            "max_steps": config.max_inference_steps if config else 0,
            "description": config.description if config else "Completed",
            "is_completed": self._current_stage == FoldingStage.COMPLETED,
        }
    
    def should_consider_stage_advance(self) -> bool:
        config = self.current_stage_config
        if config is None:
            return False
        return self._stage_step_count >= config.max_inference_steps // 2
    
    def get_stage_one_hot(self) -> np.ndarray:
        one_hot = np.zeros(4, dtype=np.float32)
        one_hot[int(self._current_stage) - 1] = 1.0
        return one_hot
    
    def get_stage_progress(self) -> float:
        config = self.current_stage_config
        if config is None:
            return 1.0
        return min(1.0, self._stage_step_count / config.max_inference_steps)


def create_multi_stage_wrapper(
    use_dummy: bool = False,
    **kwargs,
) -> MultiStageSADPGWrapper:
    """
    Factory function to create multi-stage wrapper.
    
    Args:
        use_dummy: If True, return dummy wrapper for testing
        **kwargs: Arguments passed to wrapper constructor
        
    Returns:
        Multi-stage SADP_G wrapper
    """
    if use_dummy:
        return DummyMultiStageSADPG(**kwargs)
    return MultiStageSADPGWrapper(**kwargs)
