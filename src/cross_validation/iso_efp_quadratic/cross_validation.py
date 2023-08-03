import os
import subprocess

experiment_dir = os.path.abspath('./src/cross_validation/iso_efp_quadratic')
num_gpus = 4
num_folds = 5
train_size = 1.
num_iso = 9
kappas = [-1.,0.,0.25,0.5,1.,2.]
betas = [0.25,0.5,1.,2.,3.,4.]
graph = 2

specs = [(k,b,kf) for k in kappas for b in betas for kf in range(num_folds)]

cmds = [['python', f'{experiment_dir}/train_script.py', str(kf), str(train_size), 
    str(num_iso), str(k), str(b), str(graph), str(num_gpus)] 
    for k, b, kf in specs]

for iso_script_cmd in cmds:
    print(f'Starting {iso_script_cmd}')
    subprocess.call(iso_script_cmd)