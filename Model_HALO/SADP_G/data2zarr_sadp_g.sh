# Examples:
# bash data2zarr_sadp_g.sh Fold_Tops 1 100

# 'task_name' e.g. Fold_Tops, Fold_Dress, Fling_Trousers, etc.
# 'stage_index' e.g. 1, 2, 3, etc.
# 'train_data_num' means number of training data, e.g. 100, 200, 300, etc.



task_name=${1}
stage_index=${2}
train_data_num=${3}
#python_path=~/isaacsim_4.5.0/python.sh
python_path=/home/simo/Documents/isaac-sim-standalone-4.5.0-linux-x86_64/python.sh

export ISAAC_PATH=$python_path

$ISAAC_PATH data2zarr_sadp_g.py ${task_name} ${stage_index} ${train_data_num}