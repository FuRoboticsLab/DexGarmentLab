from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

# load external package
import os
import sys
import time
import numpy as np
import open3d as o3d
from termcolor import cprint
import threading

# load isaac-relevant package
import omni.replicator.core as rep
import isaacsim.core.utils.prims as prims_utils
from pxr import UsdGeom,UsdPhysics,PhysxSchema, Gf
from isaacsim.core.api import World
from isaacsim.core.api import SimulationContext
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid
from isaacsim.core.utils.prims import is_prim_path_valid, set_prim_visibility
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
from isaacsim.core.prims import SingleXFormPrim, SingleClothPrim, SingleRigidPrim, SingleGeometryPrim, SingleParticleSystem, SingleDeformablePrim
from isaacsim.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
from isaacsim.core.utils.types import ArticulationAction, ArticulationActions
from omni.physx.scripts import deformableUtils,particleUtils,physicsUtils

# load custom package
sys.path.append(os.getcwd())
from Env_StandAlone.BaseEnv import BaseEnv
from Env_Config.Garment.Particle_Garment import Particle_Garment
from Env_Config.Garment.Deformable_Garment import Deformable_Garment
from Env_Config.Robot.BimanualDex_Ur10e import Bimanual_Ur10e
from Env_Config.Camera.Recording_Camera import Recording_Camera
from Env_Config.Room.Real_Ground import Real_Ground
from Env_Config.Utils_Project.Code_Tools import get_unique_filename, normalize_columns
from Env_Config.Utils_Project.Parse import parse_args_val
from Env_Config.Utils_Project.Flatten_Judge import judge_fling
from Env_Config.Room.Object_Tools import set_prim_visible_group, delete_prim_group
from Model_HALO.GAM.GAM_Encapsulation import GAM_Encapsulation
from Model_HALO.SADP_G.SADP_G import SADP_G

class FlingTrousers_Env(BaseEnv):
    def __init__(
        self, 
        pos:np.ndarray=None, 
        ori:np.ndarray=None, 
        usd_path:str=None, 
        ground_material_usd:str=None,
        record_video_flag:bool=False, 
        training_data_num:int=100,
        stage_1_checkpoint_num:int=1500, 
        stage_2_checkpoint_num:int=1500, 
        stage_3_checkpoint_num:int=1500, 
    ):
        # load BaseEnv
        super().__init__()
        
        # ------------------------------------ #
        # ---        Add Env Assets        --- #
        # ------------------------------------ #
        # add ground
        self.ground = Real_Ground(
            self.scene, 
            visual_material_usd = ground_material_usd,
            # you can use materials in 'Assets/Material/Floor' to change the texture of ground.
        )
        
        # load garment
        self.garment = Particle_Garment(
            self.world, 
            pos=np.array([0, 3.0, 0.6]),
            ori=np.array([0, 0, 0]),
            usd_path="Assets/Garment/Trousers/Long/PL_LongPants016/PL_LongPants016_obj.usd" if usd_path is None else usd_path,
            # contact_offset=0.012,             
            # rest_offset=0.010,                
            # particle_contact_offset=0.012,    
            # fluid_rest_offset=0.010,
            # solid_rest_offset=0.010,
        )
        # Here are some example garments you can try:
        # "Assets/Garment/Trousers/Long/PL_LongPants016/PL_LongPants016_obj.usd"
        # "Assets/Garment/Trousers/Long/PL_091/PL_091_obj.usd"
        # "Assets/Garment/Trousers/Long/PL_Pants045/PL_Pants045_obj.usd"
        # "Assets/Garment/Trousers/Short/PS_004/PS_004_obj.usd"
        
        self.judge_garment=Particle_Garment(
            self.world,
            pos=np.array([0., 9., 0.2]),
            ori=np.array([0., 0., 0.]),
            usd_path=self.garment.usd_path,
        )
        self.judge_camera=Recording_Camera(
            camera_position=np.array([0.0, 10, 6.75]), 
            camera_orientation=np.array([0, 90.0, 90.0]),
            prim_path="/World/judge_camera",
        )
        self.garment_pcd = None
        self.points_affordance_feature = None
    
        # load bimanual_dex
        self.bimanual_dex = Bimanual_Ur10e(
            self.world,
            dexleft_pos=np.array([-0.8, 0.0, 0.5]),
            dexleft_ori=np.array([0.0, 0.0, 0.0]),
            dexright_pos=np.array([0.8, 0.0, 0.5]),
            dexright_ori=np.array([0.0, 0.0, 0.0]),
        )
        
        # load camera
        self.garment_camera = Recording_Camera(
            camera_position=np.array([0.0, 1.0, 6.75]), 
            camera_orientation=np.array([0, 90.0, 90.0]),
            prim_path="/World/garment_camera",
        )
        
        self.env_camera = Recording_Camera(
            camera_position=np.array([0.0, 5.0, 8.0]),
            camera_orientation=np.array([0, 60, -90.0]),
            prim_path="/World/env_camera",
        )
        
        # load GAM Model
        self.model = GAM_Encapsulation(catogory="Trousers")    
        
        self.sadp_g_stage_1 = SADP_G(task_name="Fling_Trousers_stage_1", data_num=training_data_num, checkpoint_num=stage_1_checkpoint_num)
          
        # ------------------------------------ #
        # --- Initialize World to be Ready --- #
        # ------------------------------------ #
        # initialize world
        self.reset()
        
        # move garment to the target position
        self.garment.set_pose(pos=np.array([pos[0], pos[1], 0.2]), ori=ori)
        self.position = [pos[0], pos[1], 0.2]
        self.orientation = ori
        
        # initialize recording camera to obtain point cloud data of garment
        self.garment_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/Garment/garment",
            ]
        )
        
        # initialize gif camera to obtain rgb with the aim of creating gif
        self.env_camera.initialize(depth_enable=True)
        
        # initialize validation_flag camera
        self.judge_camera.initialize()
        
        # add thread and record gif Asynchronously(use to collect rgb data for generating gif)
        if record_video_flag:
            self.thread_record = threading.Thread(target=self.env_camera.collect_rgb_graph_for_video)
            self.thread_record.daemon = True
        
        # open hand to be initial state
        self.bimanual_dex.set_both_hand_state("open", "open")
        
        for i in range(100):
            self.step()    
            
        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])
        cprint(f"usd_path: {usd_path}", "magenta")
        cprint(f"pos_x: {pos[0]}", "magenta")
        cprint(f"pos_y: {pos[1]}", "magenta")
        cprint(f"angle: {ori[0]}", "magenta")
        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])

        
        cprint("World Ready!", "green", "on_green")
        
        

def FlingTrousers(pos, ori, usd_path, ground_material_usd, validation_flag, record_video_flag, training_data_num, stage_1_checkpoint_num, stage_2_checkpoint_num, stage_3_checkpoint_num):
    
    env = FlingTrousers_Env(pos, ori, usd_path, ground_material_usd, record_video_flag, training_data_num, stage_1_checkpoint_num, stage_2_checkpoint_num, stage_3_checkpoint_num)
    
    if record_video_flag:
        env.thread_record.start()      

    image_judge=env.judge_camera.get_rgb_graph()
    
    # hide prim to get garment point cloud
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight"],
        visible=False,
    )
    for i in range(50):
        env.step()
        
    pcd, color = env.garment_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        real_time_watch=False,
    )
    env.garment_pcd=pcd
    
    # unhide
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight"],
        visible=True,
    )
    for i in range(50):
        env.step()

    # get manipulation points from GAM Model
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=pcd, index_list=[597, 899])

    env.points_affordance_feature = normalize_columns(points_similarity.T)

    manipulation_points[:, 2] -= 0.02
    
    env.garment.particle_material.set_gravity_scale(0.7)
    
    # move both dexhand to the manipulation points
    env.bimanual_dex.dense_move_both_ik(left_pos=manipulation_points[0], left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=manipulation_points[1], right_ori=np.array([0.406, -0.406, -0.579, 0.579]))

    for i in range(20):
        env.step()
    
    for i in range(12):
        
        print(f"Stage_1_Step: {i}")

        joint_pos_L = env.bimanual_dex.dexleft.get_joint_positions()
        joint_pos_R = env.bimanual_dex.dexright.get_joint_positions()
        joint_state = np.concatenate([joint_pos_L, joint_pos_R])
        
        obs = dict()
        obs['agent_pos']=joint_state
        obs['environment_point_cloud']=env.env_camera.get_pointcloud_from_depth()
        obs['garment_point_cloud']=env.garment_pcd
        obs['points_affordance_feature']=env.points_affordance_feature

        action=env.sadp_g_stage_1.get_action(obs)
        
        print("action_shape:",action.shape)
        
        for j in range(4):
            
            action_L = ArticulationAction(joint_positions=action[j][:30])
            action_R = ArticulationAction(joint_positions=action[j][30:])

            env.bimanual_dex.dexleft.apply_action(action_L)
            env.bimanual_dex.dexright.apply_action(action_R)
            
            for _ in range(5):    
                env.step()
                
            joint_pos_L = env.bimanual_dex.dexleft.get_joint_positions()
            joint_pos_R = env.bimanual_dex.dexright.get_joint_positions()
            joint_state = np.concatenate([joint_pos_L, joint_pos_R])
            
            obs = dict()
            obs['agent_pos']=joint_state
            obs['environment_point_cloud']=env.env_camera.get_pointcloud_from_depth()
            obs['garment_point_cloud']=env.garment_pcd
            obs['points_affordance_feature']=env.points_affordance_feature
            
            env.sadp_g_stage_1.update_obs(obs)
    
    env.garment.particle_material.set_gravity_scale(10.0)
    
    for i in range(100):
        env.step()
    
    env.garment.particle_material.set_gravity_scale(1.0)
    
       
    dexleft_prim = prims_utils.get_prim_at_path("/World/DexLeft")
    dexright_prim = prims_utils.get_prim_at_path("/World/DexRight")
    set_prim_visibility(dexleft_prim, False)
    set_prim_visibility(dexright_prim, False)
    
    for i in range(50):
        env.step()

    # if you wanna create gif, use this code. Need Cooperation with thread.
    if record_video_flag:
        if not os.path.exists("Data/Fling_Trousers_Validation_HALO/video"):
            os.makedirs("Data/Fling_Trousers_Validation_HALO/video")
        env.env_camera.create_mp4(get_unique_filename("Data/Fling_Trousers_Validation_HALO/video/video", ".mp4"))
        
    success=True
    image_end = env.garment_camera.get_rgb_graph()
    success=judge_fling(image_judge,image_end,threshold=0.2)
    cprint(f"final result: {success}", color="green", on_color="on_green")

    if validation_flag:
        if not os.path.exists("Data/Fling_Trousers_Validation_HALO"):
            os.makedirs("Data/Fling_Trousers_Validation_HALO")
        # write into .log file
        with open("Data/Fling_Trousers_Validation_HALO/validation_log.txt", "a") as f:
            f.write(f"result:{success}  usd_path:{env.garment.usd_path}  pos_x:{pos[0]}  pos_y:{pos[1]}  ori_angle:{ori[0]}\n")
        if not os.path.exists("Data/Fling_Trousers_Validation_HALO/final_state_pic"):
            os.makedirs("Data/Fling_Trousers_Validation_HALO/final_state_pic")
        env.env_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename("Data/Fling_Trousers_Validation_HALO/final_state_pic/img",".png"))
   
   
     
if __name__=="__main__":
    
    args=parse_args_val()
    
    # initial setting
    pos = np.array([0.0, 0.5, 0.2])
    ori = np.array([65.0, 0.0, 0.0])
    usd_path = None
    
    if args.garment_random_flag:
        np.random.seed(int(time.time()))
        x = np.random.uniform(-0.1, 0.1) # changeable
        y = np.random.uniform(0.5, 0.7) # changeable
        angle = np.random.uniform(65.0, 80.0)
        pos = np.array([x,y,0.0])
        ori = np.array([angle, 0.0, 0.0])
        Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assets_lists = os.path.join(Base_dir,"Model_HALO/GAM/checkpoints/Trousers/assets_list.txt")
        assets_list = []
        with open(assets_lists,"r",encoding='utf-8') as f:
            for line in f:
                clean_line = line.rstrip('\n')
                assets_list.append(clean_line)
        usd_path=np.random.choice(assets_list)
    
    FlingTrousers(pos, ori, usd_path, args.ground_material_usd, args.validation_flag, args.record_video_flag, args.training_data_num, args.stage_1_checkpoint_num, args.stage_2_checkpoint_num, args.stage_3_checkpoint_num)

    if args.validation_flag:
        simulation_app.close()
    else:
        while simulation_app.is_running():
            simulation_app.update()
    
simulation_app.close()