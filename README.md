# DexGarmentLab
cloned from DexGarmentLab but RL implementation and customized

original Readme: 

<h2 align="center">
  <b><tt>DexGarmentLab</tt>: <br>
  Dexterous Garment Manipulation Environment with <br>
  Generalizable Policy</b>
</h2>

<div align="center" margin-bottom="6em">
<b>NeurIPS 2025 Spotlight</b>
</div>

<br>

<div align="center">
    <a href="https://arxiv.org/pdf/2505.11032" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-green" alt="Paper arXiv"></a>
    <a href="https://wayrise.github.io/DexGarmentLab/" target="_blank">
    <img src="https://img.shields.io/badge/Page-DexGarmentLab-red" alt="Project Page"/></a>
    <a href="https://github.com/wayrise/DexGarmentLab" target="_blank">
    <img src="https://img.shields.io/badge/Code-Github-blue" alt="Github Code"/></a>
    <a href="https://huggingface.co/datasets/wayrise/DexGarmentLab/tree/main" target="_blank">
    <img src="https://img.shields.io/badge/Data-HuggingFace-yellow" alt="HuggingFace Data"/></a>
</div>

<br>

![](Repo_Image/Teaser.jpg)

**DexGarmentLab** includes three major components:
- **Environment**: We propose <u>Dexterous Garment Manipulation Environment</u> with 15 different task scenes (especially for bimanual coordination) based on 2500+ garments.
- **Automated Data Collection**: Because of the same structure of category-level garment, category-level generalization is accessible, which empowers our proposed <u>Automated Data Collection Pipeline</u> to handle different position, deformation and shapes of garment with task config (including grasp position and task sequence) and grasp hand pose provided by single expert demonstration.
- **Generalizable Policy**: With diverse collected demonstration data, we introduce <u> **H**ierarchical g**A**rment manipu**L**ation p**O**licy (**HALO**) </u>, combining affordance points and trajectories to generalize across different attributes in different tasks.

## 📢 MileStone

- [x] *(2025.04.25)* DexGarmentLab **Simulation Environment** Release ! 

- [x] *(2025.04.25)* DexGarmentLab **Automated Data Collection Pipeline** Release ! 

- [x] *(2025.05.09)* DexGarmentLab **Baselines and Generalizable Policy** Release !

- [x] *(2025.05.09)* DexGarmentLab **Policy Validation Environment** Release !

- [x] *(2025.05.10)* DexGarmentLab **Dataset of Garment Manipulation Tasks** Release !



## 📖 Usage

**1. IsaacSim Download**

DexGarmentLab is built upon **IsaacSim 4.5.0**, please refer to [NVIDIA Official Document](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html) for download. 

```We recommend placing the Isaac Sim source folder at `~/isaacsim_4.5.0` to match the Python interpreter path specified in the `.vscode/settings.json` file we provide. If you prefer to use a custom location, please make sure that the Python interpreter path in `.vscode/settings.json` is updated accordingly.```

We will use **~/isaacsim_4.5.0/python.sh** to run the isaacsim's python file. To facilitate the running, we can define a new alias in '.bashrc' file.

```bash
echo 'alias isaac="~/isaacsim_4.5.0/python.sh"' >> ~/.bashrc
source ~/.bashrc
```
**2. Pull Repo**

```bash
git clone git@github.com:wayrise/DexGarmentLab.git
```

**3. Project Assets Download**

Download ***Robots***, **LeapMotion**, ***Scene***, ***Garment*** directory from [huggingface](https://huggingface.co/datasets/wayrise/DexGarmentLab/tree/main).

We provide automated download script in the **Assets** directory.

Following the instructions, you can download all the assets.

```bash
isaac Assets/assets_download.py
unzip Robots.zip -d ./Assets
unzip LeapMotion.zip -d ./Assets
unzip Scene.zip -d ./Assets
unzip Garment.zip -d ./Assets
unzip Human.zip -d ./Assets
```

**4. Additional Environment Dependencies for Project**

```bash
isaac -m pip install -r requirements.txt
```




## 🏕️ Simulation Environment

![](Repo_Image/Benchmark.jpg)

We introduce 15 garment manipulation tasks across 8 categories, encompassing: 

- **Garment-Self-Interaction Task**: ```Fling Tops```, ```Fling Dress```, ```Fling Trousers```, ```Fold Tops```, ```Fold Dress```, ```Fold Trousers```. The key variables include **garment position**, **orientation**, and **shape**.

- **Garment-Environment-Interaction Task**: ```Hang Dress```, ```Hang Tops```, ```Hang Trousers```, ```Hang Coat```, ```Wear Scarf```, ```Wear Bowl Hat```, ```Wear Baseball Cap```, ```Wear Glove```, ```Store Tops```. The key variables include **garment position**, **garment orientation**, **garment shape** and **environment-interaction assets positions** (e.g., hangers, pothooks, humans, etc.)

you can run python files in 'Env_StandAlone' using following commands:

```bash
# e.g. Fixed Garment Shape, Position, Orientation and Environment Assets Position
isaac Env_StandAlone/Hang_Coat_Env.py

# There are some args you can choose
# 1. --env_random_flag : 
#   True/False, Whether enable environment randomization (including position)
#   This flag only work when task belongs to Garment-Environment-Interaction Task
# 2. --garment_random_flag: 
#   True/False, Whether enable garment randomization (including position, orientation, shape)
# 3. --record_video_flag: 
#   True/False, Whether record whole-procedure video.
# 4. --data_collection_flag: 
#   True/False, Whether collect data (for policy training).

# e.g.
isaac Env_StandAlone/Hang_Coat_Env.py --env_random_flag True --garment_random_flag True 
# means in Hang_Coat_Env, enable environment and garment randomization and execute the program.
```

## ⚒️ Automated Data Collection

Autually our data collection procedure has been embedded into **Env_StandAlone/<Task_Name>_Env.py** mentioned above. The only required step is to set **--data_collection_flag** to **True**.

We provide **Data_Collection.sh** for convenience:

```bash
# usage template: bash Data_Collection.sh <task_name> <demo_num>
# e.g.
bash Data_Collection.sh Hang_Coat 10

# 10 pieces of data will be saved into 'Data/Hang_Coat'.
# including:
# - final_state_pic: .png file, picture of final garment state, used for manual verification of task success.
# - train_data: .npz file, used for training data storage.
# - video: .mp4 file, recording whole-procedure video.
# - data_collection_log.txt: recording data collection result,  corresponding assets and task configurations.
```

You can also download our prepared data from [huggingface](https://huggingface.co/datasets/wayrise/DexGarmentLab/tree/main) and unzip them into **Data** folder. The file structure should be like:

```
Data/
├── Hang_Coat/
│   │   ├── final_state_pic
│   │   ├── train_data
│   │   ├── video
│   │   └── data_collection_log.txt
......
├── Fling_Dress/
│   │   ├── final_state_pic
│   │   ├── train_data
│   │   ├── video
│   │   └── data_collection_log.txt
```

we provide data-download script for convenience:

```bash
isaac Data/data_download.py
# after download, please unzip them into Data/
```



## 🚀 Generalizable Policy

Our policy **HALO** consists: 
- **Garment Affordance Model (GAM)**, which is used to generate target manipulation points for robot's movement. The corrsponding affordance map will also be used as denosing condition for SADP.
- **Structure-Aware Diffusion Policy (SADP)**, which is used to generate robot's subsequent movement aware of garment's structure after moving to the target manipulation points.

They can be found all in **'Model_HALO/'** directory.

### GAM

The file structure of GAM is as follows:

```
GAM/
├── checkpoints/    # checkpoints of trained GAM for different category garment
    ├──Tops_LongSleeve/     # garment category
        ├──assets_list.txt           # list of assets used for validation
        ├──assets_training_list.txt  # list of assets used for training
        ├──checkpoint.pth            # trained model
        ├──demo_garment.ply          # demo garment point cloud
    ......
    ├──Trousers/
├── model                   # meta files of GAM
├── GAM_Encapsulation.py    # encapsulation of GAM
```

For the detailed use of GAM, please refer to [GAM_Usage.md](https://github.com/wayrise/DexGarmentLab/blob/main/GAM_Usage.md). The files in **'Env_StandAlone/'** also provide example of how to use GAM.

### SADP

SADP is suitable for **Garment-Environment-Interaction tasks**. All the related tasks only have one stage.

1. **Installation**
```bash
cd Model_HALO/SADP

isaac -m pip install -e .
```

2. **Data Preparation**

We need to pre-process *.npz* data collected in **'Data/'** to *.zarr* data for training. 

The only thing you need to do is just runing '*data2zarr_sadp.sh*' in 'Model_HALO/SADP'.

```bash
cd Model_HALO/SADP

# usage template: 
# bash data2zarr_sadp.sh <task_name> <stage_index> <train_data_num>
bash data2zarr_sadp.sh Hang_Coat 1 100

# Detailed parameters information can be found in the 'data2zarr_sadp.sh' file
```

The processed data will be saved in 'Model_HALO/SADP/data'. If you wanna train SADP in your headless service, please move the data to the same position.

3. **Training**

```bash
cd Model_HALO/SADP

# usage template: 
# python train.py <task_name> <expert_data_num> <seed> <gpu_id> <DEBUG_flag>
bash train.sh Hang_Coat_stage_1 100 42 0 False

# Detailed parameters information can be found in the 'train.sh' file
# Before training, we recommend you to set DEBUG_flag to True to check the training process.
```

The checkpoints will be saved in 'Model_HALO/SADP/checkpoints'.

### SADP_G

SADP_G is suitable for **Garment-Self-Interaction tasks**, which means the denosing conditions exclude interaction-object point cloud. **Fold_Tops** and **Fold_Dress** have three stages. **Fold_Trousers**, **Fling_Dress**, **Fling_Tops** have two stages. **Fling_Trousers** only have one stage.

All the procedure are the same as SADP.

1. **Installation**
```bash
cd Model_HALO/SADP_G

isaac -m pip install -e .
```

2. **Data Preparation**
```bash
cd Model_HALO/SADP

# usage template: 
# bash data2zarr_sadp_g.sh <task_name> <stage_index> <train_data_num>
bash data2zarr_sadp_g.sh Fold_Tops 2 100

# Detailed parameters information can be found in the 'data2zarr_sadp_g.sh' file
```

3. **Training**

```bash
cd Model_HALO/SADP_G

# usage template: 
# python train.py <task_name> <expert_data_num> <seed> <gpu_id> <DEBUG_flag>
bash train.sh Fold_Tops_stage_2 100 42 0 False

# Detailed parameters information can be found in the 'train.sh' file
# Before training, we recommend you to set DEBUG_flag to True to check the training process.
```

## 🎯 IL_BASELINES

Here support two IL baselines: **Diffusion Policy**, **Diffusion Policy 3D**. Their usages are the same as SADP.

### Diffusion Policy

1. Installation

```bash
cd IL_Baselines/Diffusion_Policy

isaac -m pip install -e .
```

2. Data Preparation

```bash

cd IL_Baselines/Diffusion_Policy

bash data2zarr_dp.sh Hang_Tops 1 100
```

3. Train

```bash

cd IL_Baselines/Diffusion_Policy

bash train.sh Hang_Tops_stage_1 100 42 0 False
```

### Diffusion Policy 3D

1. Installation

```bash
cd IL_Baselines/Diffusion_Policy_3D

isaac -m pip install -e .
```

2. Data Preparation

```bash
cd IL_Baselines/Diffusion_Policy_3D

bash data2zarr_dp3.sh Hang_Dress 1 100
```

3. Training

```bash
cd IL_Baselines/Diffusion_Policy_3D

bash train.sh Hang_Dress_stage_1 100 42 0 False
```

## 🪄 Policy Validation

We provide HALO Validation file for all the tasks in 'Env_Validation/' folder. We provide 'Validation.sh' to validate the policy for different tasks.

```bash
# usage template:
# bash Validation.sh <task_name> <validation_num> <training_data_num>
bash Validation.sh Hang_Coat 100 100

# Detailed parameters information can be found in the 'Validation.sh' file
```

You can find how to load checkpoints and validate the policy through the files in 'Env_Validation/' folder and we summarize core code in [Validation_Core.md](https://github.com/wayrise/DexGarmentLab/blob/main/Validation_Core.md).



## 🔐 Task Extension

Based on our simulation environment, there are lots of tasks that can be extended. If you want to add a new task, you can follow the following steps:

1. Define **task sequence** and organize **task assets** on your own.

2. Define **demo grasp points** for GAM's reference. The usage of GAM can be found in [GAM_Usage.md](https://github.com/wayrise/DexGarmentLab/blob/main/GAM_Usage.md).

3. Define **demo hand grasp pose**. We provide **LeapMotion** Solution for generating hand grasp pose through teleoperation. The usage of LeapMotion Solution can be found in [LeapMotion_Guidance.md](https://github.com/wayrise/DexGarmentLab/blob/main/LeapMotion_Guidance.md). The Guidance procedure has been demonstrated in ubuntu 20.04 and 22.04.

    After installing LeapMotion, you can run 'TeleOp_Env.py' to teleoperate ShadowHand and get hand grasp pose:

    `isaac TeleOp_Env.py`

>You only need to use right hand in real world for teleoperating both hands in simulation, because the joints of left hand and right hand are symmetric. The corresponding joint states will be printed in terminal. You can copy the joint states and paste it into **'Env_Config/Robot/BimanualDex_Ur10e.py'**.

4. Refer to python files in 'Env_StandAlone/' for the implementation of new task.


## Citation
If you find this repository useful in your research, please consider staring 🌟 this repo and citing 📑 our paper:

```
@misc{wang2025dexgarmentlab,
    title={DexGarmentLab: Dexterous Garment Manipulation Environment with Generalizable Policy},
    author={Yuran Wang and Ruihai Wu and Yue Chen and Jiarui Wang and Jiaqi Liang and Ziyu Zhu and Haoran Geng and Jitendra Malik and Pieter Abbeel and Hao Dong},
    year={2025},
    eprint={2505.11032},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
    url={https://arxiv.org/abs/2505.11032},
}
```

RL Component: 

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PPO TRAINING LOOP                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│  │   OBSERVE   │────▶│    MODEL    │────▶│    ACT      │────▶│   REWARD    │   │
│  │             │     │  (Policy)   │     │             │     │             │   │
│  │  - pcd      │     │             │     │  - delta_L  │     │  - fold     │   │
│  │  - joints   │     │  Neural     │     │  - delta_R  │     │  - compact  │   │
│  │  - ee_pose  │     │  Network    │     │  - grip_L   │     │  - height   │   │
│  │  - gam_kpts │     │             │     │  - grip_R   │     │  - action   │   │
│  └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘   │
│         ▲                                       │                   │          │
│         │                                       │                   │          │
│         │              ENVIRONMENT              │                   │          │
│         │         ┌─────────────────────────────▼───────────────────┘          │
│         │         │                                                            │
│         │         │  Isaac Sim + DexGarmentLab                                 │
│         │         │  - Particle cloth physics                                  │
│         │         │  - Robot IK control                                        │
│         │         │  - Point cloud from camera                                 │
│         │         │                                                            │
│         └─────────┴────────────────────────────────────────────────────────────│
│                                                                                 │
│  Repeat for total_timesteps (e.g., 500,000 times)                              │
│  Every n_steps (256), update policy with collected experience                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                      CUSTOM FEATURE EXTRACTOR + POLICY                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  INPUTS (Observation Dict)                                                      │
│  ─────────────────────────                                                      │
│                                                                                 │
│  garment_pcd ─────┬─▶ [Flatten] ─▶ [2048×3 → 512] ─▶ [512 → 128] ─┐            │
│  (2048, 3)        │                    ReLU              ReLU       │            │
│                   │                                                 │            │
│  joint_positions ─┼─▶ [60 → 64] ─▶ [64 → 32] ───────────────────────┤            │
│  (60,)            │      ReLU          ReLU                         │            │
│                   │                                                 │            │
│  ee_poses ────────┼─▶ [14 → 32] ─▶ [32 → 16] ───────────────────────┤            │
│  (14,)            │      ReLU          ReLU                         │            │
│                   │                                           CONCAT│            │
│  gam_keypoints ───┴─▶ [Flatten] ─▶ [18 → 32] ─▶ [32 → 16] ──────────┤            │
│  (6, 3)                               ReLU          ReLU            │            │
│                                                                     ▼            │
│                                                              ┌──────────┐        │
│                                                              │  CONCAT  │        │
│                                                              │128+32+16 │        │
│                                                              │   +16    │        │
│                                                              │  = 192   │        │
│                                                              └────┬─────┘        │
│                                                                   │              │
│                                                                   ▼              │
│                                                         [192 → 256] ReLU         │
│                                                         (Combined Features)      │
│                                                                   │              │
│                         ┌─────────────────────────────────────────┴───────┐      │
│                         │                                                 │      │
│                         ▼                                                 ▼      │
│               ┌─────────────────────┐                      ┌──────────────────┐  │
│               │    POLICY HEAD      │                      │   VALUE HEAD     │  │
│               │   (Actor - π)       │                      │   (Critic - V)   │  │
│               │                     │                      │                  │  │
│               │ [256→128] ReLU      │                      │ [256→128] ReLU   │  │
│               │ [128→64]  ReLU      │                      │ [128→64]  ReLU   │  │
│               │ [64→8] (actions)    │                      │ [64→1] (value)   │  │
│               └─────────┬───────────┘                      └────────┬─────────┘  │
│                         │                                           │            │
│                         ▼                                           ▼            │
│               ┌─────────────────────┐                      ┌──────────────────┐  │
│               │  ACTION OUTPUTS     │                      │  STATE VALUE     │  │
│               │                     │                      │                  │  │
│               │  μ (mean) + σ (std) │                      │  V(s) ≈ expected │  │
│               │  for 8D Gaussian    │                      │  future reward   │  │
│               └─────────────────────┘                      └──────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ACTION EXECUTION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Policy Output: action = [0.3, -0.2, 0.5, -0.1, 0.4, 0.2, 0.8, 0.3]            │
│                          ├─────────────┤ ├─────────────┤  │    │               │
│                          Left deltas    Right deltas   Close Open              │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ Scale by action_scale (0.05m)                                            │  │
│  │ left_delta = [0.3, -0.2, 0.5] × 0.05 = [0.015, -0.01, 0.025] meters     │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│                                      ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ Add to current EE position                                               │  │
│  │ current_left = [0.5, 0.8, 0.3]                                           │  │
│  │ target_left  = [0.5+0.015, 0.8-0.01, 0.3+0.025] = [0.515, 0.79, 0.325]  │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│                                      ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ IK Solver computes joint angles                                          │  │
│  │ target_pos [0.515, 0.79, 0.325] → joint_angles [θ₁, θ₂, θ₃, θ₄, θ₅, θ₆] │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│                                      ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ Apply to robot articulation + Step physics 5 times                       │  │
│  │ Robot arm moves, hands grasp cloth, cloth physics simulates              │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           REWARD FUNCTION BREAKDOWN                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  TOTAL REWARD = fold_progress + compactness + height_penalty + action_penalty  │
│                 + success_bonus                                                 │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1. FOLD PROGRESS (weight: 1.0)                                                │
│  ─────────────────────────────                                                 │
│  Measures reduction in XY footprint of garment                                 │
│                                                                                 │
│  ┌─────────────────┐     ┌─────────────────┐                                   │
│  │                 │     │                 │                                   │
│  │   INITIAL       │     │   FOLDED        │                                   │
│  │   ┌─────────┐   │     │   ┌────┐        │                                   │
│  │   │ garment │   │ ──▶ │   │    │        │                                   │
│  │   │  area   │   │     │   └────┘        │                                   │
│  │   │ = 0.4m² │   │     │   = 0.2m²       │                                   │
│  │   └─────────┘   │     │                 │                                   │
│  └─────────────────┘     └─────────────────┘                                   │
│                                                                                 │
│  initial_xy_area = initial_width × initial_height                              │
│  current_xy_area = current_width × current_height                              │
│  fold_progress = (initial - current) / initial                                 │
│                = (0.4 - 0.2) / 0.4 = 0.5 (50% folded)                          │
│  fold_reward = 0.5 × 1.0 = +0.5                                                │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  2. COMPACTNESS (weight: 0.5)                                                  │
│  ─────────────────────────────                                                 │
│  Measures reduction in 3D bounding box volume                                  │
│                                                                                 │
│  initial_volume = width × height × depth = 0.4 × 0.5 × 0.1 = 0.02 m³          │
│  current_volume = 0.2 × 0.3 × 0.08 = 0.0048 m³                                │
│  compactness = 1.0 - (current / initial) = 1.0 - (0.0048/0.02) = 0.76         │
│  compactness_reward = 0.76 × 0.5 = +0.38                                       │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  3. HEIGHT PENALTY (weight: 0.3)                                               │
│  ─────────────────────────────                                                 │
│  Penalizes garment "bunching up" (should stay flat)                            │
│                                                                                 │
│                    Bad: bunched up              Good: flat                      │
│                    ┌────┐                       ════════                        │
│                    │ /\ │ height_var = 0.05     height_var = 0.001             │
│                    └────┘                                                       │
│                                                                                 │
│  height_var = variance(all_points[:, z])                                       │
│  height_penalty = -height_var × 0.3 = -0.05 × 0.3 = -0.015                    │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  4. ACTION PENALTY (weight: 0.01)                                              │
│  ─────────────────────────────────                                             │
│  Encourages smooth, small movements (prevents jerky behavior)                  │
│                                                                                 │
│  action = [0.3, -0.2, 0.5, -0.1, 0.4, 0.2, 0.8, 0.3]                          │
│  action_magnitude = 0.3² + 0.2² + 0.5² + 0.1² + 0.4² + 0.2² = 0.59            │
│  action_penalty = -0.59 × 0.01 = -0.0059                                       │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  5. SUCCESS BONUS (weight: 10.0)                                               │
│  ─────────────────────────────────                                             │
│  Large reward when task is completed                                           │
│                                                                                 │
│  Success criteria:                                                             │
│    - x_ratio < 0.5 (width reduced by half)                                     │
│    - y_ratio < 0.7 (length reduced by 30%)                                     │
│    - height_var < 0.02 (garment is flat)                                       │
│                                                                                 │
│  If all criteria met: success_bonus = +10.0                                    │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  EXAMPLE CALCULATION:                                                          │
│  ────────────────────                                                          │
│  fold_progress    = +0.50                                                      │
│  compactness      = +0.38                                                      │
│  height_penalty   = -0.015                                                     │
│  action_penalty   = -0.006                                                     │
│  success_bonus    = +0.00 (not yet successful)                                 │
│  ─────────────────────────                                                     │
│  TOTAL REWARD     = +0.859                                                     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘