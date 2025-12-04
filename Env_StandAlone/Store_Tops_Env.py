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
from Env_Config.Room.Object_Tools import pusher_loader, set_prim_visible_group, delete_prim_group
from Env_Config.Utils_Project.Code_Tools import get_unique_filename, normalize_columns
from Env_Config.Utils_Project.Parse import parse_args_record
from Env_Config.Utils_Project.Collision_Group import CollisionGroup
from Env_Config.Utils_Project.Point_Cloud_Manip import rotate_point_cloud
from Model_HALO.GAM.GAM_Encapsulation import GAM_Encapsulation

class StoreTops_Env(BaseEnv):
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
            usd_path=os.getcwd() + "/" + "Assets/Garment/Tops/Collar_noSleeve_FrontClose/TCNC_Top338/TCNC_Top338_obj.usd" if usd_path is None else usd_path,
        )
        
        # load bimanual_dex
        self.bimanual_dex = Bimanual_Ur10e(
            self.world,
            dexleft_pos=np.array([-0.9, 0.0, 0.5]),
            dexleft_ori=np.array([0.0, 0.0, 0.0]),
            dexright_pos=np.array([0.9, 0.0, 0.5]),
            dexright_ori=np.array([0.0, 0.0, 0.0]),
        )

        # load camera
        self.garment_camera = Recording_Camera(
            camera_position=np.array([pos[0], pos[1], 6.75]), 
            camera_orientation=np.array([0, 90.0, 90.0]),
            prim_path="/World/garment_camera",
        )
        
        self.env_camera = Recording_Camera(
            camera_position=np.array([0.0, 5.22, 8.11]),
            camera_orientation=np.array([0, 60, -90.0]),
            prim_path="/World/env_camera",
        )
        
        self.judge_camera = Recording_Camera(
            camera_position=np.array([0.0+env_dx,1.25+env_dy,6.75]),
            camera_orientation=np.array([0, 90, -90.0]),
            prim_path="/World/judge_camera",
        )
        self.env_dx = env_dx
        self.env_dy = env_dy
        self.pusher = pusher_loader(self.scene)
        
        # load GAM Model
        self.model = GAM_Encapsulation(catogory="Tops_NoSleeve")  
        
        # define collision group - helper path
        self.helper_path=['/World/defaultGroundPlane/GroundPlane', '/World/pusher']
        self.collisiongroup = CollisionGroup(
            self.world,
            helper_path=self.helper_path,
            garment=True,
            collide_with_garment=True,
            collide_with_robot=False,
        )
        
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
        
        self.garment.set_pose(pos=np.array([pos[0], pos[1], 0.20]), ori=ori)
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
        
        self.judge_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/Garment/garment",
            ]
        )
        
        self.object_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/pusher",
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


def StoreTops(pos, ori, usd_path, env_dx, env_dy, ground_material_usd, data_collection_flag, record_video_flag):
    
    env = StoreTops_Env(pos, ori, usd_path, env_dx, env_dy, ground_material_usd, record_video_flag)
      
    # hide prim to get garment point cloud
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/pusher"],
        visible=False,
    )
    for i in range(50):
        env.step()
         
    env.garment_pcd, color = env.garment_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
    )
    
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/pusher"],
        visible=True,
    )
    for i in range(50):
        env.step()
    
    if record_video_flag:
        env.thread_record.start()
            
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=env.garment_pcd, index_list=[1954, 1832, 528, 587]) 
    manipulation_points[:, 2] = 0.025  # set z-axis to 0.005 to make sure dexhand can grasp the garment
    
    env.points_affordance_feature = normalize_columns(points_similarity[2:4].T)
    
    # move both dexhand to the manipulation points
    env.bimanual_dex.dense_move_both_ik(left_pos=manipulation_points[0], left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=manipulation_points[1], right_ori=np.array([0.406, -0.406, -0.579, 0.579]))

    env.bimanual_dex.set_both_hand_state(left_hand_state="open", right_hand_state="open")
    
    env.bimanual_dex.set_both_hand_state(left_hand_state="close", right_hand_state="close")
    
    for i in range(20):
        env.step()
    
    left_dis=np.sqrt((manipulation_points[0][0]-manipulation_points[2][0])**2+(manipulation_points[0][1]-manipulation_points[2][1])**2)
    right_dis=np.sqrt((manipulation_points[1][0]-manipulation_points[3][0])**2+(manipulation_points[1][1]-manipulation_points[3][1])**2)
    distance=(left_dis+right_dis)/4
    # get lift points
    y_off=0.01
    z_off=0.005
    left_lift_points,right_lift_points=np.array([manipulation_points[0][0], manipulation_points[0][1]-distance+y_off, distance+z_off]), np.array([manipulation_points[1][0], manipulation_points[1][1]-distance+y_off, distance+z_off]) 
    
    # move both dexhand to the lift points
    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=right_lift_points, right_ori=np.array([0.406, -0.406, -0.579, 0.579]))

    left_lift_points,right_lift_points=np.array([manipulation_points[0][0], manipulation_points[2][1]+0.06, distance+z_off]), np.array([manipulation_points[1][0], manipulation_points[3][1]+0.06, distance+z_off])
    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=right_lift_points, right_ori=np.array([0.406, -0.406, -0.579, 0.579]))

    # release the garment
    env.bimanual_dex.set_both_hand_state(left_hand_state="open", right_hand_state="open")
        
    env.garment.particle_material.set_gravity_scale(10.0)
    
    for i in range(200):
        env.step()
    
    env.garment.particle_material.set_gravity_scale(1.0)
    
    cprint("Store World Fold Procedure Finish! Store Procedure Begins!", "green", "on_green")
    
    pusher_center = np.array([0.0+env_dx, 1.10+env_dy, 0.0])
    env.pusher.set_world_pose(position=pusher_center)

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
        
    # # hide prim to get garment point cloud
    # set_prim_visible_group(
    #     prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/pusher"],
    #     visible=False,
    # )
    # for i in range(50):
    #     env.step()
         
    # env.garment_pcd, color = env.garment_camera.get_point_cloud_data_from_segment(
    #     save_or_not=False,
    #     save_path=get_unique_filename("data", extension=".ply"),
    # )
    
    # set_prim_visible_group(
    #     prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/pusher"],
    #     visible=True,
    # )
    # for i in range(50):
    #     env.step()
    
    left_ori=np.array([0.7010574,0.5609855, 0.4304593, 0.092296])
    right_ori=np.array([ 0.4304593, 0.092296, 0.7010574,0.5609855])    
    left_lift_points,right_lift_points=np.array([-0.5, 0.6, 0.65]), np.array([0.5, 0.6, 0.65])
    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=left_ori, right_pos=right_lift_points, right_ori=right_ori)
    # env.bimanual_dex.set_both_hand_state(left_hand_state="smooth", right_hand_state="smooth")

    for i in range(50):
        env.step()        
    
    garment_fold_length = np.max([
        abs(manipulation_points[0][1] - manipulation_points[2][1]),
        abs(manipulation_points[1][1] - manipulation_points[3][1])
    ]) / 2

    garment_fold_width = np.max([
        abs(manipulation_points[0][0] - manipulation_points[1][0]),
        abs(manipulation_points[2][0] - manipulation_points[3][0])
    ])

    
    manipulation_points=manipulation_points[2:]
    left_off0=np.array([-0.08,-0.0,0.005])
    right_off0=np.array([0.08,-0.0,0.005])
    left_off1=np.array([-0.05,-0.0,0.002])
    right_off1=np.array([0.05,-0.0,0.002])
    left_off2=np.array([0.005,0.05,-0.0])
    right_off2=np.array([-0.005,0.05,-0.0])
    manipulation_points[:,2]=0.00
    
    env.bimanual_dex.dense_move_both_ik(left_pos=manipulation_points[0]+left_off0, left_ori=left_ori, right_pos=manipulation_points[1]+right_off0, right_ori=right_ori)
    for i in range(20):
        env.step()
    env.bimanual_dex.dense_move_both_ik(left_pos=manipulation_points[0]+left_off1, left_ori=left_ori, right_pos=manipulation_points[1]+right_off1, right_ori=right_ori)
    for i in range(20):
        env.step()
    env.bimanual_dex.dense_move_both_ik(left_pos=manipulation_points[0]+left_off2, left_ori=left_ori, right_pos=manipulation_points[1]+right_off2, right_ori=right_ori)
    for i in range(20):
        env.step()
    # env.bimanual_dex.dense_move_both_ik(left_pos=manipulation_points[0]+left_off3, left_ori=left_ori, right_pos=manipulation_points[1]+right_off3, right_ori=right_ori)
    # for i in range(20):
    #     env.step()
    
    if data_collection_flag:
        env.record(task_name="Store_Tops",stage_index=1)    
        
    env.bimanual_dex.set_both_hand_state(left_hand_state="close", right_hand_state="close")
    
    left_point = np.array([pusher_center[0]-garment_fold_width/1.75, manipulation_points[0][1]+0.35, np.max([garment_fold_length+0.2,0.5])])
    right_point = np.array([pusher_center[0]+garment_fold_width/1.75, manipulation_points[0][1]+0.35, np.max([garment_fold_length+0.2,0.5])])
    env.bimanual_dex.dense_move_both_ik(left_pos=left_point, left_ori=left_ori, right_pos=right_point, right_ori=right_ori)
    
    left_point = np.array([pusher_center[0]-garment_fold_width/1.75, pusher_center[1], 0.5])
    right_point = np.array([pusher_center[0]+garment_fold_width/1.75, pusher_center[1], 0.5])
    env.bimanual_dex.dense_move_both_ik(left_pos=left_point, left_ori=left_ori, right_pos=right_point, right_ori=right_ori)
    
    left_point = np.array([pusher_center[0]-garment_fold_width/1.75, pusher_center[1], pusher_center[2]+0.10])
    right_point = np.array([pusher_center[0]+garment_fold_width/1.75, pusher_center[1], pusher_center[2]+0.10])
    env.bimanual_dex.dense_move_both_ik(left_pos=left_point, left_ori=left_ori, right_pos=right_point, right_ori=right_ori)
    
    # left_point = np.array([pusher_center[0]-garment_fold_width/1.35, pusher_center[1]-garment_fold_length/1.2, pusher_center[2]+0.05])
    # right_point = np.array([pusher_center[0]+garment_fold_width/1.35, pusher_center[1]-garment_fold_length/1.2, pusher_center[2]+0.05])    
    # env.bimanual_dex.dense_move_both_ik(left_pos=left_point, left_ori=left_ori, right_pos=right_point, right_ori=right_ori)
    
    env.bimanual_dex.set_both_hand_state(
        left_hand_state='open', 
        right_hand_state='open'
    )    

    if data_collection_flag:
        env.stop_record()
    
    delete_prim("/World/DexLeft")
    delete_prim("/World/DexRight")    
    for i in range(50):
        env.step()
        
    success=True
    store_state, color = env.judge_camera.get_point_cloud_data_from_segment(save_or_not=False)
    # get max_x min_x max_y min_y
    max_x = np.max(store_state[:, 0])
    min_x = np.min(store_state[:, 0])
    max_y = np.max(store_state[:, 1])
    min_y = np.min(store_state[:, 1])
    # get the center of the point cloud
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    # get the distance between the center and the pusher
    distance = np.sqrt((center_x - pusher_center[0]) ** 2 + (center_y - pusher_center[1]) ** 2)
    # if the distance is less than 0.05, it is considered as success
    if distance < 0.1:
        success=True
    else:
        success=False
        
    cprint("----------- Judge Begin -----------", "blue", attrs=["bold"])
    cprint(f"garment_center: {center_x}, {center_y}", "blue")
    cprint(f"pusher_center: {pusher_center[0]}, {pusher_center[1]}", "blue")
    cprint(f"distance: {distance}", "blue")
    cprint("----------- Judge End -----------", "blue", attrs=["bold"])
    cprint(f"final result: {success}", color="green", on_color="on_green")
    
    # if you wanna create gif, use this code. Need Cooperation with thread.
    if record_video_flag and success:
        if not os.path.exists("Data/Store_Tops/video"):
            os.makedirs("Data/Store_Tops/video")
        env.env_camera.create_mp4(get_unique_filename("Data/Store_Tops/video/video", ".mp4"))

    if data_collection_flag:
        # write into .log file
        with open("Data/Store_Tops/data_collection_log.txt", "a") as f:
            f.write(f"result:{success}  usd_path:{env.garment.usd_path}  pos_x:{pos[0]}  pos_y:{pos[1]}  env_dx:{env_dx}  env_dy:{env_dy} \n")

    if data_collection_flag:
        if success:
            env.record_to_npz(env_change=True)
            if not os.path.exists("Data/Store_Tops/final_state_pic"):
                os.makedirs("Data/Store_Tops/final_state_pic")
            env.env_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename("Data/Store_Tops/final_state_pic/img",".png"))

        

     
if __name__=="__main__":
    
    args = parse_args_record()
    
    # initial setting
    pos = np.array([0.0, 0.7, 0.20])
    ori = np.array([0.0, 0.0, 0.0])
    usd_path = None
    env_dx = 0.0
    env_dy = 0.0

    if args.env_random_flag or args.garment_random_flag:
        np.random.seed(int(time.time()))
        if args.env_random_flag:
            env_dx = np.random.uniform(-0.3, 0.3) # changeable
            env_dy = np.random.uniform(-0.1, 0.1) # changeable
        if args.garment_random_flag:
            x = np.random.uniform(-0.05, 0.05) # changeable
            y = np.random.uniform(0.65, 0.75) # changeable
            pos = np.array([x,y,0.0])
            ori = np.array([0.0, 0.0, 0.0])
            Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            assets_lists = os.path.join(Base_dir,"Model_HALO/GAM/checkpoints/Tops_NoSleeve/assets_training_list.txt")
            assets_list = []
            with open(assets_lists,"r",encoding='utf-8') as f:
                for line in f:
                    clean_line = line.rstrip('\n')
                    assets_list.append(clean_line)
            usd_path=os.getcwd() + "/" + np.random.choice(assets_list)

    StoreTops(pos, ori, usd_path, env_dx, env_dy, args.ground_material_usd, args.data_collection_flag, args.record_video_flag)

    if args.data_collection_flag:
        simulation_app.close()
    else:
        while simulation_app.is_running():
            simulation_app.update()

simulation_app.close()