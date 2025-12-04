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
from isaacsim.core.utils.prims import is_prim_path_valid, set_prim_visibility
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
from isaacsim.core.prims import SingleXFormPrim, SingleClothPrim, SingleRigidPrim, SingleGeometryPrim, SingleParticleSystem, SingleDeformablePrim
from isaacsim.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
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
from Env_Config.Utils_Project.Parse import parse_args_record
from Env_Config.Utils_Project.Flatten_Judge import judge_fling
from Env_Config.Room.Object_Tools import set_prim_visible_group, delete_prim_group
from Model_HALO.GAM.GAM_Encapsulation import GAM_Encapsulation

class FlingTops_Env(BaseEnv):
    def __init__(
        self, 
        pos:np.ndarray=None, 
        ori:np.ndarray=None, 
        usd_path:str=None, 
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
            usd_path=os.getcwd() + "/" + "Assets/Garment/Tops/NoCollar_Lsleeve_FrontClose/TNLC_Top089/TNLC_Top089_obj.usd" if usd_path is None else usd_path,
        )

        # if collect data, load judge_camera and judge_garment to data_collection_flag the success of final state
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
        
        # load GAM
        self.model = GAM_Encapsulation(catogory="Tops_LongSleeve")        
        
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
        
        # initialize data_collection_flag camera
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
    
    def record_callback(self, step_size):

        if self.step_num % 5 == 0:
        
            joint_pos_L = self.bimanual_dex.dexleft.get_joint_positions()
            
            joint_pos_R = self.bimanual_dex.dexright.get_joint_positions()
            
            joint_state = np.array([*joint_pos_L, *joint_pos_R])

            rgb = self.env_camera.get_rgb_graph(save_or_not=False)

            point_cloud = self.env_camera.get_pointcloud_from_depth(
                show_original_pc_online=False,
                sample_flag=True,
                sampled_point_num=2048,
                show_downsample_pc_online=False,
            )
            
            self.saving_data.append({ 
                "joint_state": joint_state,
                "image": rgb,
                "env_point_cloud": point_cloud,
                "garment_point_cloud":self.garment_pcd,
                "points_affordance_feature": self.points_affordance_feature,
            })
        
        self.step_num += 1
     
def FlingTops(pos, ori, usd_path, ground_material_usd, data_collection_flag, record_video_flag):

    env = FlingTops_Env(pos, ori, usd_path, ground_material_usd, record_video_flag)

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
    
    # get manipulation points from GAM
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=pcd, index_list=[838,179]) 
    
    env.points_affordance_feature = normalize_columns(points_similarity.T)
    
    manipulation_points[:, 2] = 0.002 
    
    # env.garment.particle_material.set_gravity_scale(0.7)
    
    # move both dexhand to the manipulation points
    env.bimanual_dex.dense_move_both_ik(left_pos=manipulation_points[0], left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=manipulation_points[1], right_ori=np.array([0.406, -0.406, -0.579, 0.579]))
        
    if data_collection_flag:
        for i in range(20):
            env.step()
        env.record(task_name="Fling_Tops", stage_index=1)
        
    env.bimanual_dex.set_both_hand_state(left_hand_state="close", right_hand_state="close")
    
    
    # get lift points
    # calculate the approximate distance of the grasping point.
    distance=np.sqrt((manipulation_points[0][0]-manipulation_points[1][0])**2+(manipulation_points[0][1]-manipulation_points[1][1])**2)/2
    
    
    left_lift_points,right_lift_points=np.array([-0.1, 0.5, 0.85]), np.array([0.1, 0.5, 0.85]) 
    
    # move both dexhand to the lift points
    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=right_lift_points, right_ori=np.array([0.406, -0.406, -0.579, 0.579]))

    left_lift_points,right_lift_points=np.array([-distance-0.02, 1.4, 0.15]), np.array([distance+0.02, 1.4, 0.15])
    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=right_lift_points, right_ori=np.array([0.406, -0.406, -0.579, 0.579]))
    
    # release the garment
    env.bimanual_dex.set_both_hand_state(left_hand_state="open", right_hand_state="open")
    
    if data_collection_flag:
        env.stop_record()
    
    env.garment.particle_material.set_gravity_scale(10.0)
    
    for i in range(100):
        env.step()
    
    env.garment.particle_material.set_gravity_scale(1.0)
    

    # move hand away to ensure the clothing is not obstructed
    left_lift_points,right_lift_points=np.array([-0.5, 1.3, 0.65]), np.array([0.5, 1.3, 0.65])
    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=right_lift_points, right_ori=np.array([0.406, -0.406, -0.579, 0.579]))
    
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

    # get manipulation points from GAM

    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=pcd, index_list=[1635,954,838,179])
    
    env.points_affordance_feature = normalize_columns(points_similarity[0:2].T)
    
    manipulation_points[:, 2] = 0.002   # set z-axis to 0.01 to make sure dexhand can grasp the garment
    
    # move both dexhand to the manipulation points
    env.bimanual_dex.dense_move_both_ik(left_pos=manipulation_points[0], left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=manipulation_points[1], right_ori=np.array([0.406, -0.406, -0.579, 0.579]))
    
    if data_collection_flag:
        for i in range(20):
            env.step()
        env.record(task_name="Fling_Tops", stage_index=2)
        
    env.bimanual_dex.set_both_hand_state(left_hand_state="close", right_hand_state="close")

    # flatten the sleeve of the garment
    left_len  =  np.sqrt((manipulation_points[0][0]-manipulation_points[2][0])**2+(manipulation_points[0][1]-manipulation_points[2][1])**2)/2
    right_len =  np.sqrt((manipulation_points[2][0]-manipulation_points[3][0])**2+(manipulation_points[1][1]-manipulation_points[3][1])**2)/2
    sleeve_len=(left_len+right_len)/2+0.06
    manipulation_points[0][0]-=sleeve_len
    manipulation_points[1][0]+=sleeve_len
    manipulation_points[:,1]=1.28
    manipulation_points[:, 2] = 0.1
    env.bimanual_dex.dense_move_both_ik(left_pos=manipulation_points[0], left_ori=np.array([0.579, -0.579, -0.406, 0.406]), right_pos=manipulation_points[1], right_ori=np.array([0.406, -0.406, -0.579, 0.579]))

    env.bimanual_dex.set_both_hand_state(left_hand_state="open", right_hand_state="open")
    
    if data_collection_flag:
        env.stop_record()
    
    env.garment.particle_material.set_gravity_scale(10.0)
    
    for i in range(150):
        env.step()
        
    env.garment.particle_material.set_gravity_scale(1.0)
    
    dexleft_prim = prims_utils.get_prim_at_path("/World/DexLeft")
    dexright_prim = prims_utils.get_prim_at_path("/World/DexRight")
    set_prim_visibility(dexleft_prim, False)
    set_prim_visibility(dexright_prim, False)
    
    for i in range(50):
        env.step()
        
    success=True
    image_end = env.garment_camera.get_rgb_graph()
    success=judge_fling(image_judge,image_end,threshold=0.2)
    cprint(f"final result: {success}", color="green", on_color="on_green")
    
    # if you wanna create gif, use this code. Need Cooperation with thread.
    if record_video_flag and success:
        if not os.path.exists("Data/Fling_Tops/video"):
            os.makedirs("Data/Fling_Tops/video")
        env.env_camera.create_mp4(get_unique_filename("Data/Fling_Tops/video/video", ".mp4"))

    if data_collection_flag:
        # write into .log file
        with open("Data/Fling_Tops/data_collection_log.txt", "a") as f:
            f.write(f"result:{success}  usd_path:{env.garment.usd_path}  pos_x:{pos[0]}  pos_y:{pos[1]}  ori_angle:{ori[0]}\n")
    
    if data_collection_flag and success:
        env.record_to_npz()
        if not os.path.exists("Data/Fling_Tops/final_state_pic"):
            os.makedirs("Data/Fling_Tops/final_state_pic")
        env.env_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename("Data/Fling_Tops/final_state_pic/img",".png"))

   
        
if __name__=="__main__":
    
    args=parse_args_record()
    
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
        assets_lists = os.path.join(Base_dir,"Model_HALO/GAM/checkpoints/Tops_LongSleeve/assets_training_list.txt")
        assets_list = []
        with open(assets_lists,"r",encoding='utf-8') as f:
            for line in f:
                clean_line = line.rstrip('\n')
                assets_list.append(clean_line)
        usd_path=os.getcwd() + "/" + np.random.choice(assets_list)
    
    FlingTops(pos, ori, usd_path, args.ground_material_usd, args.data_collection_flag, args.record_video_flag)

    if args.data_collection_flag:
        simulation_app.close()
    else:
        while simulation_app.is_running():
            simulation_app.update()
    
simulation_app.close()