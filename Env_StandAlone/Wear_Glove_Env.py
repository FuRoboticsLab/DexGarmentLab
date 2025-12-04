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
from Env_Config.Utils_Project.Code_Tools import get_unique_filename
from Env_Config.Utils_Project.Parse import parse_args_record
from Env_Config.Utils_Project.Point_Cloud_Manip import rotate_point_cloud
from Model_HALO.GAM.GAM_Encapsulation import GAM_Encapsulation
from Env_Config.Utils_Project.Collision_Group import CollisionGroup
from Env_Config.Human.Human import Human
from Env_Config.Utils_Project.Attachment_Block import attach_fixedblock,AttachmentBlock
# from Env_Config.Garment.Deformable_Garment import DeformableConfig

class WearGlove_Env(BaseEnv):
    def __init__(
        self, 
        pos:np.ndarray=None, 
        ori:np.ndarray=None, 
        usd_path:str=None, 
        env_dx:float=0.0,
        env_dy:float=0.0,
        ground_material_usd:str=None,
        data_collection_flag:bool=False,
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
            path="Assets/Human/human_model.usd",
            position=[0.0+env_dx,2.3+env_dy,-0.84], 
            orientation=[90.0,0.0,90.0],
        )
        self.human_center = np.array([-0.012+env_dx, 1.65+env_dy, 0.135])
        
        self.garment=Deformable_Garment(
            self.world,
            usd_path=os.getcwd() + "/" + "Assets/Garment/Glove/GL_Gloves068/GL_Gloves068_obj.usd" if usd_path is None else usd_path,
            pos=np.array([0, 3.0, 0.6]),
            ori=np.array([0.0, 0.0, 0.0]),
            scale=np.array([0.01,0.01,0.01]),
            youngs_modulus=1e7
        )


        self.deformable_path_group=[]
        self.deformable_path_group.append(self.garment.deformable_prim_path)

        self.attach=AttachmentBlock(self.world,garment_path=self.deformable_path_group)
        self.attach.create_block(block_name="block1",block_position=[-3.0, 3.0, 3.0],block_visible=False)
        self.attach.create_block(block_name="block2",block_position=[3.0, 3.0, 3.0],block_visible=False)
        self.attach.enable_disable_gravity(0,False)
        self.attach.enable_disable_gravity(1,False)
        self.attach_path=self.attach.block_path_list

        self.scene.add(
            FixedCuboid(
                name="hanger_helper",
                position=[0.0,1.24,0.05],
                prim_path="/World/hanger_helper",
                scale=np.array([1.0,1.0,0.1]),
                orientation=euler_angles_to_quat([0.0,0.0,0.0],degrees=True),
                color=np.array([180,180,180]),
                visible=True,
            )
        )

        self.helper_path=["/World/hanger_helper", "/World/Human"]

        self.collisiongroup=CollisionGroup(self.world,helper_path=self.helper_path,garment=False,collide_with_robot=False,collide_with_garment=True)     
        self.collisiongroup.add_collision(group_path="attach",target=self.attach_path)



        # load bimanual_dex
        self.bimanual_dex = Bimanual_Ur10e(
            self.world,
            dexleft_pos=np.array([-0.6, 0.7, 0.5]),
            dexleft_ori=np.array([0.0, 0.0, 0.0]),
            dexright_pos=np.array([0.6, 0.7, 0.5]),
            dexright_ori=np.array([0.0, 0.0, 0.0]),
        )

        # load camera
        self.garment_camera = Recording_Camera(
            camera_position=np.array([pos[0], pos[1], 6.75]), 
            camera_orientation=np.array([0, 90.0, 90.0]),
            prim_path="/World/garment_camera",
        )
        
        
        self.env_camera = Recording_Camera(
            camera_position=np.array([0, 9.55, 9.8]),
            camera_orientation=np.array([0, 50.0, -90.0]),
            prim_path="/World/env_camera",
        )
        
        # load UniGarmentManip Model
        self.model = GAM_Encapsulation(catogory="Glove")  
        
        self.garment_pcd = None
        self.left_key_point_feature = None
        self.right_key_point_feature = None

        # ------------------------------------ #
        # --- Initialize World to be Ready --- #
        # ------------------------------------ #
        # initialize world
        self.reset()
        
        self.garment.set_garment_pose(pos=np.array([pos[0], pos[1], 0.2]), ori=ori)
        self.garment.set_mass(0.02)
        
        # initialize recording camera to obtain point cloud data of garment
        self.garment_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/Deformable/deformable",
            ]
        )
        # initialize gif camera to obtain rgb with the aim of creating gif
        self.env_camera.initialize(
            depth_enable=True,
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
        

def WearGlove(pos, ori, usd_path, env_dx, env_dy, ground_material_usd, data_collection_flag, record_video_flag):

    env = WearGlove_Env(pos, ori, usd_path, env_dx, env_dy, ground_material_usd, data_collection_flag, record_video_flag)
        
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
        real_time_watch=False,
    )
    
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight", "/World/Human"],
        visible=True,
    )
    for i in range(50):
        env.step()
        
    if record_video_flag:
        env.thread_record.start()
        
    # pcd_rotate = rotate_point_cloud(env.garment_pcd, euler_angles=np.array([0, 0, 180]), center_point=env.garment.get_garment_center_pos())     

    # # get manipulation points from UniGarmentManip Model
    # manipulation_points, indices, point_features = env.model.get_manipulation_points(input_pcd=pcd_rotate, index_list=[70, 464])
    # manipulation_points = env.garment_pcd[indices]    

    # manipulation_points[:, 2] = 0.125

    # env.left_key_point_feature = point_features[0]
    # env.right_key_point_feature = point_features[1]
    
    min_y = np.min(env.garment_pcd[:, 1])
    max_y = np.max(env.garment_pcd[:, 1])
    glove_length = max_y - min_y
    
    threshold = 0.02  # 允许在 max_y ± 0.01 范围内
    mask = (np.abs(env.garment_pcd[:, 1] - max_y) < threshold) & (env.garment_pcd[:, 2] > 0.12)
    subset = env.garment_pcd[mask]
    
    point_x_max = subset[np.argmax(subset[:, 0])]
    point_x_min = subset[np.argmin(subset[:, 0])]
    
    point_x_max[2] = 0.12
    point_x_min[2] = 0.12
    
    env.attach.set_block_position(0, point_x_max)
    env.attach.set_block_position(1, point_x_min)   
    
    for i in range(50):
        env.step() 
    
    env.attach.attach([0,1])
    

    

    # env.thread_record.start()     # if you want to record gif, please uncomment this line


    pos_left=env.attach.get_block_position(1)[0]
    pos_right=env.attach.get_block_position(0)[0]


    pos_left[0] -= 0.045
    pos_right[0] += 0.045

    env.bimanual_dex.dense_move_both_ik(left_pos=pos_left, left_ori=np.array([-0.129, 0.837, 0.483, -0.224]), right_pos=pos_right, right_ori=np.array([0.483, -0.224, -0.129, 0.837]))

    # pos_left[0] += 0.025
    # pos_right[0] -= 0.025

    # # while simulation_app.is_running():
    # #     env.step()

    env.bimanual_dex.set_both_hand_state(left_hand_state="open", right_hand_state="open")
    
    env.bimanual_dex.set_both_hand_state(left_hand_state="pinch", right_hand_state="pinch")

    left_lift_point = np.array([env.human_center[0]-0.15, env.human_center[1], env.human_center[2]-0.02])
    right_lift_point = np.array([env.human_center[0]+0.15, env.human_center[1], env.human_center[2]-0.02])

    env.bimanual_dex.move_both_with_blocks(left_pos=left_lift_point, left_ori=np.array([-0.129, 0.837, 0.483, -0.224]), right_pos=right_lift_point, right_ori=np.array([0.483, -0.224, -0.129, 0.837]),attach=env.attach,indices=[0,1])


    left_lift_point = np.array([env.human_center[0]-0.15, env.human_center[1]+glove_length*1.15, env.human_center[2]])
    right_lift_point = np.array([env.human_center[0]+0.15, env.human_center[1]+glove_length*1.15, env.human_center[2]])

    env.bimanual_dex.move_both_with_blocks(left_pos=left_lift_point, left_ori=np.array([-0.129, 0.837, 0.483, -0.224]), right_pos=right_lift_point, right_ori=np.array([0.483, -0.224, -0.129, 0.837]),attach=env.attach,indices=[0,1])


    env.bimanual_dex.set_both_hand_state(left_hand_state="open", right_hand_state="open")
    
    left_lift_points=np.array([-0.8,1.5,0.3])
    right_lift_points=np.array([0.8,1.5,0.3])

    env.bimanual_dex.dense_move_both_ik(left_pos=left_lift_points, left_ori=np.array([0.0,0.866,0.500,0.0]), right_pos=right_lift_points, right_ori=np.array([0.500,0.0,0.0,0.866]))

    env.attach.detach(0)
    env.attach.detach(1)
    
    for i in range(100):
        env.step()
    
    if record_video_flag:
            if not os.path.exists("Data/Wear_Glove/video"):
                os.makedirs("Data/Wear_Glove/video")
            env.env_camera.create_mp4(get_unique_filename("Data/Wear_Glove/video/video", ".mp4"))

 

if __name__=="__main__":
    
    args = parse_args_record()
    
    # initial setting
    pos = np.array([0.0, 1.25, 0.15])
    ori = np.array([0.0, 90.0, 0.0])
    usd_path = None
    env_dx = 0.0
    env_dy = 0.0

    WearGlove(pos, ori, usd_path, env_dx, env_dy, args.ground_material_usd, args.data_collection_flag, args.record_video_flag)

    if args.data_collection_flag:
        simulation_app.close()
    else:
        while simulation_app.is_running():
            simulation_app.update()

simulation_app.close()