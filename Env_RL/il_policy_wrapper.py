"""
IL Policy Wrapper for Residual RL.

This module provides a unified interface to load and query pre-trained
IL policies (SADP, DP3, Diffusion Policy) without modifying their original code.

The wrapper handles:
- Loading pretrained checkpoints
- Converting observations to the format expected by IL policies
- Getting action predictions
- Managing observation history (for temporal models)

Usage:
    wrapper = ILPolicyWrapper(
        policy_type="SADP",
        task_name="Fold_Tops",
        checkpoint_num=1000,
        data_num=100,
    )
    il_action = wrapper.get_action(observation)
"""

import os
import sys
import numpy as np
import torch
from typing import Dict, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PolicyType(Enum):
    """Supported IL policy types."""
    SADP = "SADP"
    SADP_G = "SADP_G"
    DP3 = "DP3"
    DIFFUSION_POLICY = "DiffusionPolicy"


@dataclass
class ILPolicyConfig:
    """Configuration for IL policy loading."""
    policy_type: PolicyType
    task_name: str
    checkpoint_num: int
    data_num: int
    device: str = "cuda:0"


class ILPolicyWrapper:
    """
    Unified wrapper for IL policies.
    
    Provides a consistent interface regardless of the underlying IL architecture.
    The IL policy is kept frozen and only used for inference.
    """
    
    def __init__(
        self,
        policy_type: Union[str, PolicyType],
        task_name: str,
        checkpoint_num: int,
        data_num: int,
        device: str = "cuda:0",
    ):
        """
        Initialize IL policy wrapper.
        
        Args:
            policy_type: Type of IL policy ("SADP", "SADP_G", "DP3")
            task_name: Name of the task (e.g., "Fold_Tops")
            checkpoint_num: Checkpoint number to load
            data_num: Data configuration number
            device: Device to run inference on
        """
        if isinstance(policy_type, str):
            policy_type = PolicyType(policy_type)
        
        self.policy_type = policy_type
        self.task_name = task_name
        self.checkpoint_num = checkpoint_num
        self.data_num = data_num
        self.device = device
        
        self._policy = None
        self._loaded = False
        
        # Action dimensions (for bimanual robot)
        # 60 DOF total: 30 per arm+hand
        self.action_dim = 60
        
        # Observation buffer for temporal models
        self._obs_history = []
        self._n_obs_steps = 8  # Default for diffusion policies
        
    def load(self):
        """Load the IL policy from checkpoint."""
        if self._loaded:
            return
        
        print(f"[ILPolicyWrapper] Loading {self.policy_type.value} policy...")
        print(f"  Task: {self.task_name}")
        print(f"  Checkpoint: {self.checkpoint_num}")
        print(f"  Data: {self.data_num}")
        
        try:
            if self.policy_type == PolicyType.SADP:
                from Model_HALO.SADP.SADP import SADP
                self._policy = SADP(
                    task_name=self.task_name,
                    checkpoint_num=self.checkpoint_num,
                    data_num=self.data_num,
                    device=self.device,
                )
                
            elif self.policy_type == PolicyType.SADP_G:
                from Model_HALO.SADP_G.SADP_G import SADP_G
                self._policy = SADP_G(
                    task_name=self.task_name,
                    checkpoint_num=self.checkpoint_num,
                    data_num=self.data_num,
                    device=self.device,
                )
                
            elif self.policy_type == PolicyType.DP3:
                from IL_Baselines.Diffusion_Policy_3D.DP3 import DP3
                self._policy = DP3(
                    task_name=self.task_name,
                    checkpoint_num=self.checkpoint_num,
                    data_num=self.data_num,
                    device=self.device,
                )
                
            else:
                raise ValueError(f"Unsupported policy type: {self.policy_type}")
            
            self._loaded = True
            print(f"[ILPolicyWrapper] Policy loaded successfully!")
            
        except Exception as e:
            print(f"[ILPolicyWrapper] Failed to load policy: {e}")
            raise
    
    def reset(self):
        """Reset observation history."""
        self._obs_history = []
        if self._loaded and hasattr(self._policy, 'env_runner'):
            self._policy.env_runner.reset_obs()
    
    def get_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get action from IL policy.
        
        Args:
            observation: Dictionary containing:
                - environment_point_cloud: (N, 3) environment point cloud
                - garment_point_cloud: (N, 3) garment point cloud  
                - object_point_cloud: (N, 3) optional object point cloud
                - points_affordance_feature: (N, 2) GAM affordance features
                - agent_pos: (60,) joint positions
                
        Returns:
            action: (n_action_steps, action_dim) or (action_dim,) action array
        """
        if not self._loaded:
            self.load()
        
        # Convert observation format if needed
        obs = self._prepare_observation(observation)
        
        # Get action from policy
        with torch.no_grad():
            action = self._policy.get_action(obs)
        
        # Ensure numpy array
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        return action
    
    def get_action_single_step(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get single-step action (first action in sequence).
        
        For diffusion policies that output action sequences, this returns
        only the first action to be executed.
        
        Args:
            observation: Observation dict
            
        Returns:
            action: (action_dim,) single action
        """
        action = self.get_action(observation)
        
        # If action is a sequence, return first
        if len(action.shape) > 1:
            return action[0]
        return action
    
    def _prepare_observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Prepare observation for IL policy.
        
        Handles format conversion and ensures required keys exist.
        """
        prepared = {}
        
        # Required keys for SADP/DP3
        required_keys = [
            "environment_point_cloud",
            "garment_point_cloud", 
            "points_affordance_feature",
            "agent_pos",
        ]
        
        # Optional keys
        optional_keys = [
            "object_point_cloud",
        ]
        
        # Copy required keys
        for key in required_keys:
            if key in obs:
                prepared[key] = obs[key]
            else:
                # Create default values
                if "point_cloud" in key:
                    prepared[key] = np.zeros((2048, 3), dtype=np.float32)
                elif key == "points_affordance_feature":
                    prepared[key] = np.zeros((2048, 2), dtype=np.float32)
                elif key == "agent_pos":
                    prepared[key] = np.zeros(self.action_dim, dtype=np.float32)
        
        # Copy optional keys if present
        for key in optional_keys:
            if key in obs:
                prepared[key] = obs[key]
            else:
                # Default to garment point cloud
                prepared[key] = prepared.get("garment_point_cloud", 
                                              np.zeros((2048, 3), dtype=np.float32))
        
        return prepared
    
    @property
    def is_loaded(self) -> bool:
        """Check if policy is loaded."""
        return self._loaded
    
    @property
    def action_dimension(self) -> int:
        """Get action dimension."""
        return self.action_dim


class DummyILPolicy:
    """
    Dummy IL policy for testing without loading actual checkpoints.
    
    Returns zero actions, useful for debugging the residual RL setup.
    """
    
    def __init__(self, action_dim: int = 60):
        self.action_dim = action_dim
        self._loaded = True
    
    def load(self):
        pass
    
    def reset(self):
        pass
    
    def get_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Return zero action."""
        return np.zeros(self.action_dim, dtype=np.float32)
    
    def get_action_single_step(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Return zero action."""
        return np.zeros(self.action_dim, dtype=np.float32)
    
    @property
    def is_loaded(self) -> bool:
        return True
    
    @property
    def action_dimension(self) -> int:
        return self.action_dim


def create_il_policy(
    policy_type: str = "dummy",
    task_name: str = "Fold_Tops",
    checkpoint_num: int = 1000,
    data_num: int = 100,
    device: str = "cuda:0",
) -> Union[ILPolicyWrapper, DummyILPolicy]:
    """
    Factory function to create IL policy.
    
    Args:
        policy_type: "SADP", "SADP_G", "DP3", or "dummy"
        task_name: Task name
        checkpoint_num: Checkpoint number
        data_num: Data configuration
        device: Device for inference
        
    Returns:
        IL policy wrapper instance
    """
    if policy_type.lower() == "dummy":
        return DummyILPolicy()
    
    return ILPolicyWrapper(
        policy_type=policy_type,
        task_name=task_name,
        checkpoint_num=checkpoint_num,
        data_num=data_num,
        device=device,
    )



