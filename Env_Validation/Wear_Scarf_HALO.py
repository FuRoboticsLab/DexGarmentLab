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
from isaacsim.core.utils.prims import is_prim_path_valid, set_prim_visibility, delete_prim
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
from isaacsim.core.prims import SingleXFormPrim, SingleClothPrim, SingleRigidPrim, SingleGeometryPrim, SingleParticleSystem, SingleDeformablePrim
from isaacsim.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
from isaacsim.core.utils.rotations import euler_angles_to_quat
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
from Env_Config.Room.Object_Tools import hat_helper_load, set_prim_visible_group, delete_prim_group
from Env_Config.Utils_Project.Code_Tools import get_unique_filename, normalize_columns
from Env_Config.Utils_Project.Parse import parse_args_val
from Env_Config.Utils_Project.Point_Cloud_Manip import rotate_point_cloud
from Model_HALO.GAM.GAM_Encapsulation import GAM_Encapsulation
from Env_Config.Utils_Project.Collision_Group import CollisionGroup
from Env_Config.Human.Human import Human
from Env_Config.Utils_Project.Attachment_Block import attach_fixedblock
from Model_HALO.SADP.SADP import SADP

class WearScarf_Env(BaseEnv):
    def __init__(
        self, 
        pos:np.ndarray=None, 
        ori:np.ndarray=None, 
        usd_path:str=None, 
        env_dx:float=0.0,
        env_dy:float=0.0,
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
        self.ground = Real_Ground(
            self.scene, 
            visual_material_usd = ground_material_usd,
            # you can use materials in 'Assets/Material/Floor' to change the texture of ground.
        )
        
        # load garment
        self.garment = Particle_Garment(
            self.world, 
            # pos=np.array([0.0, 0.25, 0.15]),
            pos=np.array([0.0, 3.0, 0.6]),
            ori=np.array([0.0, 0.0, 0.0]),
            usd_path="Assets/Flatten_Scarf/flatten_scarf_0.4.usd" if usd_path is None else usd_path,
            friction=1.0,
            particle_adhesion_scale=1.0,
            particle_friction_scale=1.0,
            contact_offset=0.015,         
            rest_offset=0.01,            
            particle_contact_offset=0.015,
            fluid_rest_offset=0.01,
            solid_rest_offset=0.01,
        )
        
        # load human
        self.env_dx = env_dx
        self.env_dy = env_dy
        self.human = Human(
            path="Assets/Human/human_model.usd",
            position=[0.0+env_dx, 0.9+env_dy, -0.7],
            orientation=[90.0, 0.0, 180.0]
        )
        self.human_center = np.array([0.0+env_dx, 0.9+env_dy, 0.7])

        # load bimanual_dex
        self.bimanual_dex = Bimanual_Ur10e(
            self.world,
            dexleft_pos=np.array([-1.05, 1.15, 0.45]),
            dexleft_ori=np.array([0.0, 0.0, -90.0]),
            dexright_pos=np.array([1.25, 1.15, 0.45]),
            dexright_ori=np.array([0.0, 0.0, 90.0]),
        )
        
        # load camera
        self.garment_camera = Recording_Camera(
            camera_position=np.array([0.0, 0.75, 8.0]), 
            camera_orientation=np.array([0, 90.0, 90.0]),
            prim_path="/World/garment_camera",
        )
        
        self.env_camera = Recording_Camera(
            camera_position=np.array([0.0, 5.4, 5.85]),
            camera_orientation=np.array([0, 50.0, -90.0]),
            prim_path="/World/env_camera",
        )
        
        self.judge_front_camera = Recording_Camera(
            camera_position=np.array([self.human_center[0], 6.32, 0.42]),
            camera_orientation=np.array([0, 0.0, -90.0]),
            prim_path="/World/judge_front_camera",
        )
        
        self.judge_back_camera = Recording_Camera(
            camera_position=np.array([self.human_center[0], -4.2, 0.36]),
            camera_orientation=np.array([0.0, 0.0, 90.0]),
            prim_path="/World/judge_back_camera",
        )
        
        # load UniGarmentManip Model
        self.model = GAM_Encapsulation(catogory="Scarf")        

        
        # helper for seperating scarf
        self.helper_1 = FixedCuboid(
            prim_path = "/World/helper/helper_1",
            color=np.array([0.0, 0.0, 1.0]),
            name = "helper_1",
            position = [0.0, 0.8, 0.0],
            scale=[0.3, 1.0, 0.3],
            orientation=[0.924, 0.0, 0.383, 0.0],
            size=1.0,
            visible = False,
        )
        # self.helper_2 = FixedCuboid(
        #     prim_path = "/World/helper/helper_2",
        #     color=np.array([0.0, 0.0, 1.0]),
        #     name = "helper_2",
        #     position = [0.45, 0.8, 0.0],
        #     scale=[0.3, 1.0, 0.3],
        #     orientation=[0.924, 0.0, 0.383, 0.0],
        #     size=1.0,
        #     visible = False,
        # )
        # self.helper_3 = FixedCuboid(
        #     prim_path = "/World/helper/helper_3",
        #     color=np.array([0.0, 0.0, 1.0]),
        #     name = "helper_3",
        #     position = [-0.45, 0.8, 0.0],
        #     scale=[0.3, 1.0, 0.3],
        #     orientation=[0.924, 0.0, 0.383, 0.0],
        #     size=1.0,
        #     visible = False,
        # )


        
        # define collision group - helper path
        self.helper_path=['/World/defaultGroundPlane/GroundPlane', '/World/Human']
        self.collisiongroup = CollisionGroup(
            self.world,
            helper_path=self.helper_path,
            garment=True,
            collide_with_garment=True,
            collide_with_robot=False,
        )
        
        self.object_camera = Recording_Camera(
            camera_position=np.array([0.0, 5.4, 5.85]),
            camera_orientation=np.array([0, 50.0, -90.0]),
            prim_path="/World/object_camera",
        )

        self.garment_pcd = None
        self.object_pcd = None
        self.points_affordance_feature = None
        
        self.sadp = SADP(task_name="Wear_Scarf_stage_1", data_num=training_data_num, checkpoint_num=stage_1_checkpoint_num)
        
        # ------------------------------------ #
        # --- Initialize World to be Ready --- #
        # ------------------------------------ #
        # initialize world
        self.reset()
        
        self.garment.set_pose(pos=np.array([pos[0], pos[1], 0.35]), ori=ori)
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
        
        self.judge_front_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/Garment/garment",
            ]
        )
        
        self.judge_back_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/Garment/garment",
            ]
        )
        
        self.object_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/Human",
            ]
        )
        
        # add thread and record gif Asynchronously(use to collect rgb data for generating gif)
        if record_video_flag:
            self.thread_record = threading.Thread(target=self.env_camera.collect_rgb_graph_for_video)
            self.thread_record.daemon = True

        # step world to make it ready
                
        for i in range(100):
            self.step()
        
        delete_prim_group(["/World/helper/helper_1"])
            
        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])
        cprint(f"usd_path: {usd_path}", "magenta")
        cprint(f"pos_x: {pos[0]}", "magenta")
        cprint(f"pos_y: {pos[1]}", "magenta")
        cprint(f"env_dx: {env_dx}", "magenta")
        cprint(f"env_dy: {env_dy}", "magenta")
        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])
        
        cprint("World Ready!", "green", "on_green")

def WearScarf(pos, ori, usd_path, env_dx, env_dy, ground_material_usd, validation_flag, record_video_flag, training_data_num, stage_1_checkpoint_num, stage_2_checkpoint_num, stage_3_checkpoint_num):
    
    env = WearScarf_Env(pos, ori, usd_path, env_dx, env_dy, ground_material_usd, record_video_flag, training_data_num, stage_1_checkpoint_num, stage_2_checkpoint_num, stage_3_checkpoint_num)

    # hide prim to get object point cloud
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/Garment/garment"],
        visible=False,
    )
    for i in range(50):
        env.step()
    
    env.object_pcd, color = env.object_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        # real_time_watch=True,
    )
    
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/Garment/garment"],
        visible=True,
    )
    for i in range(50):
        env.step()
        
    # hide prim to get garment point cloud
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/Human"],
        visible=False,
    )
    for i in range(50):
        env.step()

    env.garment_pcd, color = env.garment_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
    )

    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/Human"],
        visible=True,
    )
    for i in range(50):
        env.step()
        
    if record_video_flag:
        env.thread_record.start()
    
    # manipulation_points, indices, max_values = env.model.get_manipulation_points(input_pcd=env.garment_pcd, index_list=[205, 908])
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=env.garment_pcd, index_list=[205, 1600])
    
    env.points_affordance_feature = normalize_columns(points_similarity.T)
        
    manipulation_points[:, 2] = -0.02
    manipulation_points[:, 1] -= 0.05

    # move to initial position
    env.bimanual_dex.dense_move_both_ik(
        left_pos=np.array([-0.5, 0.5, 0.5]),
        left_ori=np.array([0.707, -0.707, 0.0, 0.0]),
        right_pos=np.array([0.5, 0.5, 0.5]),
        right_ori=np.array([0.0, 0.0, -0.707, 0.707]),
    )
    
    env.bimanual_dex.set_both_hand_state(
        left_hand_state='open', 
        right_hand_state='open'
    )

    # move to grasp point
    env.bimanual_dex.dense_move_both_ik(
        left_pos=manipulation_points[0],
        left_ori=np.array([0.707, -0.707, 0.0, 0.0]),
        right_pos=manipulation_points[1],
        right_ori=np.array([0.0, 0.0, -0.707, 0.707]),
    )
    
    env.garment.particle_material.set_gravity_scale(0.5) 
    
    for i in range(20):
        env.step()
    
    for i in range(24):
        
        print(f"Stage_1_Step: {i}")

        joint_pos_L = env.bimanual_dex.dexleft.get_joint_positions()
        joint_pos_R = env.bimanual_dex.dexright.get_joint_positions()
        joint_state = np.concatenate([joint_pos_L, joint_pos_R])
        
        obs = dict()
        obs['agent_pos']=joint_state
        obs['environment_point_cloud']=env.env_camera.get_pointcloud_from_depth()
        obs['garment_point_cloud']=env.garment_pcd
        obs['object_point_cloud']=env.object_pcd
        obs['points_affordance_feature']=env.points_affordance_feature

        action=env.sadp.get_action(obs)
        
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
            obs['object_point_cloud']=env.object_pcd
            obs['points_affordance_feature']=env.points_affordance_feature
            
            env.sadp.update_obs(obs)
            
    for i in range(50):
        env.step()
        
    # if you wanna create gif, use this code. Need Cooperation with thread.
    if record_video_flag:
        if not os.path.exists("Data/Wear_Scarf_Validation_HALO/video"):
            os.makedirs("Data/Wear_Scarf_Validation_HALO/video")
        env.env_camera.create_mp4(get_unique_filename("Data/Wear_Scarf_Validation_HALO/video/video", ".mp4"))
        
    set_prim_visible_group(["/World/DexLeft", "/World/DexRight", "/World/Human"], visible=False)
        
    for i in range(50):
        env.step()
        
    success=True
    scarf_front, color = env.judge_front_camera.get_point_cloud_data_from_segment(save_or_not=False)
    scarf_back, color = env.judge_back_camera.get_point_cloud_data_from_segment(save_or_not=False)
    
    # calculate sum
    front_points_below_threshold = np.sum(scarf_front[:, 2] < 0.02)
    back_points_below_threshold = np.sum(scarf_back[:, 2] < 0.02)
    
    if front_points_below_threshold < 20 and back_points_below_threshold < 20:
        success = True
    else:
        success = False
        
    set_prim_visible_group(["/World/Human"], visible=True)
    for i in range(50):
        env.step()
    
    cprint("----------- Judge Begin -----------", "blue", attrs=["bold"])
    cprint(f"front_points_below_threshold: {front_points_below_threshold}", "blue")
    cprint(f"back_points_below_threshold: {back_points_below_threshold}", "blue")
    cprint("----------- Judge End -----------", "blue", attrs=["bold"])
    cprint(f"final result: {success}", color="green", on_color="on_green")
       

    if validation_flag:
        if not os.path.exists("Data/Wear_Scarf_Validation_HALO"):
            os.makedirs("Data/Wear_Scarf_Validation_HALO")
        # write into .log file
        with open("Data/Wear_Scarf_Validation_HALO/validation_log.txt", "a") as f:
            f.write(f"result:{success}  usd_path:{env.garment.usd_path}  pos_x:{pos[0]}  pos_y:{pos[1]}  env_dx:{env_dx}  env_dy:{env_dy} \n")
        if not os.path.exists("Data/Wear_Scarf_Validation_HALO/final_state_pic"):
            os.makedirs("Data/Wear_Scarf_Validation_HALO/final_state_pic")
        env.env_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename("Data/Wear_Scarf_Validation_HALO/final_state_pic/img",".png"))
      
if __name__=="__main__":
    
    args = parse_args_val()
    
    # initial setting
    pos = np.array([0.0, 0.30, 0.65])
    ori = np.array([90.0, 0.0, 0.0])
    usd_path = None
    env_dx = 0.0
    env_dy = 0.0

    if args.env_random_flag or args.garment_random_flag:
        np.random.seed(int(time.time()))
        if args.env_random_flag:
            env_dx = np.random.uniform(-0.05, 0.1) # changeable
            env_dy = np.random.uniform(-0.05, 0.05) # changeable
        if args.garment_random_flag:
            x = np.random.uniform(-0.05, 0.05) # changeable
            y = np.random.uniform(0.30, 0.40) # changeable
            pos = np.array([x,y,0.0])
            ori = np.array([90.0, 0.0, 0.0])
            Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            assets_lists = os.path.join(Base_dir,"Model_HALO/GAM/checkpoints/Scarf/assets_list.txt")
            assets_list = []
            with open(assets_lists,"r",encoding='utf-8') as f:
                for line in f:
                    clean_line = line.rstrip('\n')
                    assets_list.append(clean_line)
            usd_path=np.random.choice(assets_list)

    WearScarf(pos, ori, usd_path, env_dx, env_dy, args.ground_material_usd, args.validation_flag, args.record_video_flag, args.training_data_num, args.stage_1_checkpoint_num, args.stage_2_checkpoint_num, args.stage_3_checkpoint_num)

    if args.validation_flag:
        simulation_app.close()
    else:
        while simulation_app.is_running():
            simulation_app.update()

simulation_app.close()