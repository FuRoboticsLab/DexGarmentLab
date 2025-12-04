from typing import Dict

from structure_aware_diffusion_policy.policy.base_policy import BasePolicy


class BaseRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePolicy) -> Dict:
        raise NotImplementedError()
