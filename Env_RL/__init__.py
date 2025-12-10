# Env_RL: Reinforcement Learning environments for DexGarmentLab
#
# Three RL approaches are available:
#
# 1. Flat RL (FoldTopsGymEnv):
#    - Direct action control (8D continuous)
#    - RL learns full control from scratch
#    - Most flexible, but hardest to train
#
# 2. Hierarchical RL (HierarchicalFoldEnv):
#    - Discrete primitive selection (6 choices)
#    - RL learns WHEN to execute primitives
#    - Low-level control handled by pre-defined primitives
#
# 3. Residual RL (ResidualFoldEnv):
#    - IL policy provides base actions (frozen)
#    - RL learns small corrections (bounded residuals)
#    - Final action = IL_action + RL_residual
#    - Best for fine-tuning existing IL policies

# Flat RL environment (direct action control)
from Env_RL.fold_tops_gym_env import FoldTopsGymEnv

# Hierarchical RL environment (primitive selection)
from Env_RL.hierarchical_fold_env import HierarchicalFoldEnv
from Env_RL.primitives import ManipulationPrimitives, PrimitiveID, PrimitiveResult

# Residual RL environment (IL + RL corrections)
from Env_RL.residual_fold_env import ResidualFoldEnv
from Env_RL.il_policy_wrapper import (
    ILPolicyWrapper,
    DummyILPolicy,
    PolicyType,
    create_il_policy,
)

__all__ = [
    # Flat RL
    "FoldTopsGymEnv",
    # Hierarchical RL
    "HierarchicalFoldEnv",
    "ManipulationPrimitives",
    "PrimitiveID",
    "PrimitiveResult",
    # Residual RL
    "ResidualFoldEnv",
    "ILPolicyWrapper",
    "DummyILPolicy",
    "PolicyType",
    "create_il_policy",
]

