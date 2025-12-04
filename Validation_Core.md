1. SADP
```python
# load package
from Model_HALO.SADP.SADP import SADP

# load model
sadp = SADP(task_name="Hang_Coat_stage_1", data_num=100, checkpoint_num=3000)

# input organization
obs = dict()
obs['agent_pos']=XXX
obs['environment_point_cloud']=XXX
obs['garment_point_cloud']=XXX
obs['object_point_cloud']=XXX
obs['points_affordance_feature']=XXX

# get action or update obs
action=sadp.get_action(obs) # or sadp.update_obs(obs)
```

2. SADP_G
```python
# load package
from Model_HALO.SADP_G.SADP_G import SADP_G

# load model
sadp_g = SADP_G(task_name="Hang_Coat_stage_1", data_num=100, checkpoint_num=3000)

# input organization
obs = dict()
obs['agent_pos']=XXX
obs['environment_point_cloud']=XXX
obs['garment_point_cloud']=XXX
obs['points_affordance_feature']=XXX

# get action or update obs
action=sadp_g.get_action(obs) # or sadp_g.update_obs(obs)
```

3. DP3
```python
# load package
from IL_Baselines.Diffusion_Policy_3D.DP3 import DP3

# load model
dp3 = DP3(task_name="Hang_Coat_stage_1", data_num=100, checkpoint_num=3000)

# input organization
obs = dict()
obs['agent_pos']=XXX
obs['point_cloud']=XXX

# get action or update obs
action=dp3.get_action(obs) # or dp3.update_obs(obs)
```

4. DP
```python
# load package
from IL_Baselines.Diffusion_Policy.DP import DP

# load model
dp = DP(task_name="Hang_Coat_stage_1", data_num=100, checkpoint_num=3000)

# input organization
obs = dict()
obs['agent_pos']=XXX
obs['head_cam']=np.moveaxis(RGB_numpy, -1, 0) / 255.0

# get action or update obs
action=dp.get_action(obs) # or dp.update_obs(obs)
```