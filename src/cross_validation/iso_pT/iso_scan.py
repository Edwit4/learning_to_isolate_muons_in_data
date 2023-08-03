import os
import subprocess
import numpy as np

experiment_dir = os.path.abspath('./src/cross_validation/iso_pT')
num_gpus = 4
num_folds = 5
train_size = 1.
min_cones = 1
max_cones = 18

cmds = [['python', f'{experiment_dir}/train_script.py', str(kf), str(train_size), 
    str(iso), str(num_gpus)] 
    for kf in range(num_folds) 
    for iso in range(min_cones,max_cones+1)]

for iso_script_cmd in cmds:
    print(f'Starting {iso_script_cmd}')
    subprocess.call(iso_script_cmd)