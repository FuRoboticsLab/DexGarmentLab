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
from Env_Config.Room.Real_Room import Real_Room
from Model_HALO.GAM.GAM_Encapsulation import GAM_Encapsulation

class RealScene_Load_Env(BaseEnv):
    def __init__(
        self, 
    ):
        # load BaseEnv
        super().__init__()
        
        # ------------------------------------ #
        # ---        Add Env Assets        --- #
        # ------------------------------------ #
        
        # add ground
        self.ground = Real_Ground(
            self.scene, 
            visual_material_usd = os.getcwd()+"/Assets/Material/Floor/Collected_WoodFloor004/WoodFloor004.usd" # os.getcwd()+"/Assets/Material/Floor/WoodFloor002/WoodFloor002.usd"
            # you can use materials in 'Assets/Material/Floor' to change the texture of ground.
        )
        
        # Real_Room(self.scene, usd_path=os.getcwd()+'/Assets/Scene/kitchen/kitchen_8/kitchen8.usd')
        
        # ------------------------------------ #
        # --- Initialize World to be Ready --- #
        # ------------------------------------ #
        # initialize world
        self.reset()
    

      
if __name__=="__main__":
    
    RealScene_Load_Env()
    
    while simulation_app.is_running():
        simulation_app.update()
    
simulation_app.close()