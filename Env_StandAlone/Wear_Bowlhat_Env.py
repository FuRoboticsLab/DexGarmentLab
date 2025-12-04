from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

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
from Env_Config.Utils_Project.Parse import parse_args_record
from Env_Config.Utils_Project.Point_Cloud_Manip import rotate_point_cloud,compute_similarity
from Model_HALO.GAM.GAM_Encapsulation import GAM_Encapsulation
from Env_Config.Utils_Project.Collision_Group import CollisionGroup
from Env_Config.Human.Human import Human
from Env_Config.Utils_Project.Attachment_Block import attach_fixedblock


class WearBowlhat_Env(BaseEnv):
    def __init__(
        self, 
        pos:np.ndarray=None, 
        ori:np.ndarray=None, 
        usd_path:str=None, 
        env_dx:float=0.0,
        env_dy:float=0.0,
        ground_material_usd:str=None,
        record_video_flag:bool=False, 
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
        
        # load human
        self.env_dx = env_dx
        self.env_dy = env_dy
        self.human = Human(
            path=os.getcwd() + "/" +"Assets/Human/Collected_human_model/biped_demo_meters.usd",
            position=[0.0+env_dx,1.15+env_dy,0.0], 
            scale=np.array([0.6, 0.6, 0.6]),
        )

        # load garment
        self.garment=Deformable_Garment(
            self.world,
            pos=np.array([0, 3.0, 0.6]),
            ori=np.array([0.0, 0.0, 0.0]),
            usd_path=os.getcwd() + "/" + "Assets/Garment/Hat/HA_Hat007/HA_Hat007_obj.usd" if usd_path is None else usd_path,
            scale=np.array([0.009,0.0075,0.0075]),
            youngs_modulus=1e6,
            dynamic_friction=25.0
        )

        self.target_put_pos = hat_helper_load(self.scene, pos[0], pos[1], self.env_dx, self.env_dy)

        self.helper_path=["/World/hanger_helper", "/World/hanger", "/World/Human"]

        self.head_helper_path=["/World/head_helper"]

        # load bimanual_dex
        self.bimanual_dex = Bimanual_Ur10e(
            self.world,
            dexleft_pos=np.array([-0.6, 0.0, 0.5]),
            dexleft_ori=np.array([0.0, 0.0, 0.0]),
            dexright_pos=np.array([0.6, 0.0, 0.5]),
            dexright_ori=np.array([0.0, 0.0, 0.0]),
        )
        
        # load camera
        self.garment_camera = Recording_Camera(
            camera_position=np.array([0.0, 0.5, 6.75]), 
            camera_orientation=np.array([0, 90.0, 90.0]),
            prim_path="/World/garment_camera",
        )
        
        self.env_camera = Recording_Camera(
            camera_position=np.array([0, -3.45, 4.17]),
            camera_orientation=np.array([0, 40, 90]),
            prim_path="/World/env_camera",
        )
        
        # import Collision Group
        self.collisiongroup=CollisionGroup(
            self.world,
            helper_path=self.helper_path,
            garment=False,
            collide_with_garment=True,
            collide_with_robot=False
        )     
        self.collisiongroup.add_collision(group_path="head",target=self.head_helper_path)

        self.object_camera = Recording_Camera(
            camera_position=np.array([0.0, -6.6, 4.9]),
            camera_orientation=np.array([0, 30.0, 90.0]),
            prim_path="/World/object_camera",
        )

        self.garment_pcd = None
        self.object_pcd = None
        self.points_affordance_feature = None

        # ------------------------------------ #
        # --- Initialize World to be Ready --- #
        # ------------------------------------ #
        # initialize world
        self.reset()

        self.garment.set_garment_pose(pos=np.array([pos[0], pos[1], 0.65]), ori=ori)
        self.position = [pos[0], pos[1], 0.2]
        self.orientation = ori
                
        # initialize recording camera to obtain point cloud data of garment
        self.garment_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/Deformable/deformable",
            ]
        )

        self.env_camera.initialize(
            depth_enable=True,
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
            
        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])
        cprint(f"usd_path: {usd_path}", "magenta")
        cprint(f"pos_x: {pos[0]}", "magenta")
        cprint(f"pos_y: {pos[1]}", "magenta")
        cprint(f"env_dx: {env_dx}", "magenta")
        cprint(f"env_dy: {env_dy}", "magenta")
        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])
        
        cprint("World Ready!", "green", "on_green")
        
    def record_callback(self, step_size):

        if self.step_num % 5 == 0:
        
            joint_pos_L = self.bimanual_dex.dexleft.get_joint_positions()
            
            joint_pos_R = self.bimanual_dex.dexright.get_joint_positions()
            
            joint_state = np.array([*joint_pos_L, *joint_pos_R])

            rgb = self.env_camera.get_rgb_graph(save_or_not=False)

            point_cloud = self.env_camera.get_pointcloud_from_depth(
                show_original_pc_online=False,
                show_downsample_pc_online=False,
            )
            
            self.saving_data.append({ 
                "joint_state": joint_state,
                "image": rgb,
                "env_point_cloud": point_cloud,
                "garment_point_cloud":self.garment_pcd,
                "object_point_cloud":self.object_pcd,
                "points_affordance_feature": self.points_affordance_feature,
            })
        
        self.step_num += 1
        
        

def WearBowlhat(pos, ori, usd_path, env_dx, env_dy, ground_material_usd, data_collection_flag, record_video_flag):
    
    env = WearBowlhat_Env(pos, ori, usd_path, env_dx, env_dy, ground_material_usd, record_video_flag)
    
    # hide prim to get object point cloud
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/Deformable/deformable", "/World/hanger"],
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
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/Deformable/deformable", "/World/hanger"],
        visible=True,
    )
    for i in range(50):
        env.step()
    
    # hide prim to get garment point cloud
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/Human", "/World/hanger"],
        visible=False,
    )
    for i in range(50):
        env.step()
    
    env.garment_pcd, color = env.garment_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
    )
    
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/Human", "/World/hanger"],
        visible=True,
    )
    for i in range(50):
        env.step()
        
    if record_video_flag:
        env.thread_record.start()
    
    center_y = env.garment.get_garment_center_pos()[1]
    mask = np.abs(env.garment_pcd[:, 1] - center_y) < 0.02
    garment_subset = env.garment_pcd[mask]
    
    if len(garment_subset) > 0:
        point_x_max = garment_subset[np.argmax(garment_subset[:, 0])]
        point_x_min = garment_subset[np.argmin(garment_subset[:, 0])]
        cprint(f"left_manipulation_points: {point_x_min}", "green")
        cprint(f"right_manipulation_points: {point_x_max}", "green")
    else:
        cprint("error!", "red", "on_red")
        simulation_app.close()
        
    left_similarity = compute_similarity(env.garment_pcd, point_x_min, sigma=0.05)
    right_similarity = compute_similarity(env.garment_pcd, point_x_max, sigma=0.05)
    
    env.points_affordance_feature = normalize_columns(np.concatenate([left_similarity, right_similarity], axis=1))
        
    hat_length = np.max(env.garment_pcd[:, 1]) - np.min(env.garment_pcd[:, 1])
    
    left_lift_points = np.array([point_x_min[0]-0.05, point_x_min[1], 0.4])
    right_lift_points = np.array([point_x_max[0]+0.05, point_x_max[1], 0.4])

    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.653,0.653,0.271,0.271]), right_pos=right_lift_points, right_ori=np.array([0.271,0.271,0.653,0.653]), dense_sample_scale=0.005)
    
    for i in range(50):
        env.step()
    
    left_lift_points = np.array([point_x_min[0], point_x_min[1], 0.56])
    right_lift_points = np.array([point_x_max[0], point_x_max[1], 0.56])   
    
    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.653,0.653,0.271,0.271]), right_pos=right_lift_points, right_ori=np.array([0.271,0.271,0.653,0.653]), dense_sample_scale=0.005)

    for i in range(50):
        env.step() 
        
    env.garment.set_mass(0.075)

    if data_collection_flag:
        for i in range(20):
            env.step()
        env.record(task_name="Wear_Bowlhat", stage_index=1)
        
    env.bimanual_dex.set_both_hand_state(left_hand_state="smooth", right_hand_state="smooth")
    
    env.bimanual_dex.set_both_hand_state(left_hand_state="close", right_hand_state="close")
    
    bias = env.target_put_pos[0] - env.garment.get_garment_center_pos()[0]
    left_lift_points = np.array([point_x_min[0]+bias, point_x_min[1], env.target_put_pos[2]])
    right_lift_points = np.array([point_x_max[0]+bias, point_x_max[1], env.target_put_pos[2]])
    
    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.653,0.653,0.271,0.271]), right_pos=right_lift_points, right_ori=np.array([0.271,0.271,0.653,0.653]))

    left_lift_points = np.array([point_x_min[0]+bias, env.target_put_pos[1]+hat_length/2, env.target_put_pos[2]])
    right_lift_points = np.array([point_x_max[0]+bias, env.target_put_pos[1]+hat_length/2, env.target_put_pos[2]])

    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.653,0.653,0.271,0.271]), right_pos=right_lift_points, right_ori=np.array([0.271,0.271,0.653,0.653]))

    left_lift_points = np.array([point_x_min[0]+bias, env.target_put_pos[1]+hat_length/2, env.target_put_pos[2]-0.3])
    right_lift_points = np.array([point_x_max[0]+bias, env.target_put_pos[1]+hat_length/2, env.target_put_pos[2]-0.3])

    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.653,0.653,0.271,0.271]), right_pos=right_lift_points, right_ori=np.array([0.271,0.271,0.653,0.653]))
        
    if data_collection_flag:
        env.stop_record()
        
    for i in range(50):
        env.step()
        
    attach_fixedblock(env.stage,env.garment.deformable_prim_path+"/mesh/attachment",env.garment.deformable_prim_path,"/World/head_helper")
 
    delete_prim_group(["/World/DexLeft", "/World/DexRight"])
        
    for i in range(50):
        env.step()
    
    success=True
    # judge successful or not
    cur_pos=env.garment.get_garment_center_pos()
    print(cur_pos)
    distance = np.linalg.norm(cur_pos-np.array([env.target_put_pos[0],env.target_put_pos[1]+0.11,1.06]))
    if distance<0.05:
        success=True
    else:
        success=False
        
    cprint("----------- Judge Begin -----------", "blue", attrs=["bold"])
    cprint(f"hat_cur_pos: {env.garment.get_garment_center_pos()}", "blue")
    cprint(f"judge_pos: {np.array([env.target_put_pos[0],env.target_put_pos[1]+0.07,1.06])}", "blue")
    cprint(f"distance between garment and head: {distance}", "blue")
    cprint("----------- Judge End -----------", "blue", attrs=["bold"])
    cprint(f"final result: {success}", color="green", on_color="on_green")
    
    # if you wanna create gif, use this code. Need Cooperation with thread.
    if record_video_flag and success:
        if not os.path.exists("Data/Wear_Bowlhat/video"):
            os.makedirs("Data/Wear_Bowlhat/video")
        env.env_camera.create_mp4(get_unique_filename("Data/Wear_Bowlhat/video/video", ".mp4"))


    if data_collection_flag:
        # write into .log file
        with open("Data/Wear_Bowlhat/data_collection_log.txt", "a") as f:
            f.write(f"result:{success}  usd_path:{env.garment.usd_path}  pos_x:{pos[0]}  pos_y:{pos[1]}  env_dx:{env_dx}  env_dy:{env_dy} \n")

    if data_collection_flag:
        if success:
            env.record_to_npz(env_change=True)
            if not os.path.exists("Data/Wear_Bowlhat/final_state_pic"):
                os.makedirs("Data/Wear_Bowlhat/final_state_pic")
            env.env_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename("Data/Wear_Bowlhat/final_state_pic/img",".png"))


if __name__=="__main__":
    
    args = parse_args_record()
    
    # initial setting
    pos = np.array([0.0, 0.8, 0.65])
    ori = np.array([90.0, 0.0, 0.0])
    usd_path = None
    env_dx = 0.0
    env_dy = 0.0 

    if args.env_random_flag or args.garment_random_flag:
        np.random.seed(int(time.time()))
        if args.env_random_flag:
            env_dx = np.random.uniform(-0.15, 0.15) # changeable
            env_dy = np.random.uniform(-0.05, 0.05) # changeable
        if args.garment_random_flag:
            x = np.random.uniform(-0.1, 0.1) # changeable
            y = np.random.uniform(0.7, 0.9) # changeable
            pos = np.array([x,y,0.0])
            ori = np.array([90.0, 0.0, 0.0])
            Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            assets_lists = os.path.join(Base_dir,"Model_HALO/GAM/checkpoints/Bowl_Hat/assets_training_list.txt")
            assets_list = []
            with open(assets_lists,"r",encoding='utf-8') as f:
                for line in f:
                    clean_line = line.rstrip('\n')
                    assets_list.append(clean_line)
            usd_path=os.getcwd() + "/" + np.random.choice(assets_list)
            print(usd_path)

    WearBowlhat(pos, ori, usd_path, env_dx, env_dy, args.ground_material_usd, args.data_collection_flag, args.record_video_flag)

    if args.data_collection_flag:
        simulation_app.close()
    else:
        while simulation_app.is_running():
            simulation_app.update()

simulation_app.close()
