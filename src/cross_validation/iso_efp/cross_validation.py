import os
import subprocess

experiment_dir = os.path.abspath('./src/cross_validation/iso_efp')
num_gpus = 4
num_folds = 5
train_size = 1.
num_iso = 9

cmds = [['python', f'{experiment_dir}/train_script.py', str(kf), str(train_size), 
    str(num_iso), str(num_gpus)] for kf in range(num_folds)]

for iso_script_cmd in cmds:
    print(f'Starting {iso_script_cmd}')
    subprocess.call(iso_script_cmd)