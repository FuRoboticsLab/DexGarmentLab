# Examples:
# bash data2zarr_sadp.sh Hang_Coat 1 100

# 'task_name' e.g. Hang_Coat, Hang_Tops, Wear_Scarf, etc.
# 'stage_index' e.g. 1, 2, 3, etc.
# 'train_data_num' means number of training data, e.g. 100, 200, 300, etc.



task_name=${1}
stage_index=${2}
train_data_num=${3}
python_path=~/isaacsim_4.5.0/python.sh

export ISAAC_PATH=$python_path

$ISAAC_PATH data2zarr_sadp.py ${task_name} ${stage_index} ${train_data_num}