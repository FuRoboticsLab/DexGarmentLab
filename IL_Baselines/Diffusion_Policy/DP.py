import sys
import os
import torch
import hydra
import dill

from IL_Baselines.Diffusion_Policy.diffusion_policy.workspace.robotworkspace import RobotWorkspace
from IL_Baselines.Diffusion_Policy.diffusion_policy.common.pytorch_util import dict_apply
from IL_Baselines.Diffusion_Policy.diffusion_policy.policy.base_image_policy import BaseImagePolicy
from IL_Baselines.Diffusion_Policy.diffusion_policy.env_runner.dp_runner import DPRunner

class DP:
    def __init__(self, task_name, checkpoint_num: int, data_num: int):
        self.policy = get_policy(f'checkpoints/{task_name}_{data_num}/{checkpoint_num}.ckpt', None, 'cuda:0')
        self.runner = DPRunner(output_dir=None)

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

    def reset_obs(self):
        self.runner.reset_obs()
    
def get_policy(checkpoint, output_dir, device):
    # load checkpoint
    payload = torch.load(open('./IL_Baselines/Diffusion_Policy/'+checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: RobotWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    return policy