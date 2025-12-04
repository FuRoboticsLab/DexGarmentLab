import argparse
import json
import os
import shutil

import numpy as np
import zarr
from PIL import Image
from tqdm import tqdm
import copy
import open3d as o3d
import pickle

def main():
    parser = argparse.ArgumentParser(
        description="Convert data to zarr format for diffusion policy"
    )
    parser.add_argument(
        "task_name",
        type=str,
        default="Fold_Dress",
        help="The name of the task (e.g., Fold_Dress)",
    )
    parser.add_argument(
        "stage_index",
        type=int,
        default=1,
        help="The index of current stage (e.g., 1)",
    )
    parser.add_argument(
        "train_data_num",
        type=int,
        default=200,
        help="Number of data to process (e.g., 200)",
    )
    args = parser.parse_args()
    
    task_name = args.task_name
    stage_index = args.stage_index
    train_data_num = args.train_data_num
    
    current_abs_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_abs_dir))
    print("Project Root Dir : ", parent_dir)
    
    load_dir = parent_dir + f"/Data/{task_name}/train_data"
    print("Meta Data Load Dir : ", load_dir)
    
    save_dir = f"data/{task_name}_stage_{stage_index}_{train_data_num}.zarr"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    print("Save Dir : ", save_dir)
    
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")
    
    # ZARR datasets will be created dynamically during the first batch write
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    
    # Batch processing settings
    batch_size = 100
    point_cloud_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    total_count = 0
    current_batch = 0
    
    for current_ep in tqdm(range(train_data_num), desc=f"Processing {train_data_num} MetaData"):
        data = np.load(load_dir + f'/data_{current_ep}.npz', allow_pickle=True)
        meta_data = data[f'stage_{stage_index}']
        data_length = len(meta_data)
        for i in range(data_length-1):
            point_cloud_arrays.append(meta_data[i]['env_point_cloud'])
            state_arrays.append(meta_data[i]['joint_state'])
            action_arrays.append(meta_data[i+1]['joint_state'])
            total_count += 1
        episode_ends_arrays.append(copy.deepcopy(total_count))
        
        # Write to ZARR if batch is full or if this is the last episode
        if (current_ep + 1) % batch_size == 0 or (current_ep + 1) == train_data_num:
            # Convert arrays to NumPy
            point_cloud_arrays = np.array(point_cloud_arrays)  # NHWC -> NCHW
            action_arrays = np.array(action_arrays)
            state_arrays = np.array(state_arrays)
            episode_ends_arrays = np.array(episode_ends_arrays)
            
            # Create datasets dynamically during the first write
            if current_batch == 0:
                zarr_data.create_dataset(
                    "point_cloud",
                    shape=(0, *point_cloud_arrays.shape[1:]),
                    chunks=(batch_size, *point_cloud_arrays.shape[1:]),
                    dtype=point_cloud_arrays.dtype,
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_data.create_dataset(
                    "state",
                    shape=(0, state_arrays.shape[1]),
                    chunks=(batch_size, state_arrays.shape[1]),
                    dtype="float32",
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_data.create_dataset(
                    "action",
                    shape=(0, action_arrays.shape[1]),
                    chunks=(batch_size, action_arrays.shape[1]),
                    dtype="float32",
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_meta.create_dataset(
                    "episode_ends",
                    shape=(0,),
                    chunks=(batch_size,),
                    dtype="int64",
                    compressor=compressor,
                    overwrite=True,
                )
            
            # Append data to ZARR datasets
            zarr_data["point_cloud"].append(point_cloud_arrays)
            zarr_data["state"].append(state_arrays)
            zarr_data["action"].append(action_arrays)
            zarr_meta["episode_ends"].append(episode_ends_arrays)
            
            print(
                f"Batch {current_batch + 1} written with {len(point_cloud_arrays)} samples."
            )

            print(f"point_cloud shape: {point_cloud_arrays.shape}")
            print(f"state shape: {state_arrays.shape}")
            print(f"action shape: {action_arrays.shape}")
            print(f"episode_ends shape: {episode_ends_arrays.shape}")

            # Clear arrays for next batch
            point_cloud_arrays = []
            action_arrays = []
            state_arrays = []
            episode_ends_arrays = []
            current_batch += 1
            

if __name__ == "__main__":
    main()