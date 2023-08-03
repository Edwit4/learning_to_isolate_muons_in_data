import os
import shutil
import argparse
import torch
import subprocess
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('starting_bootstrap', metavar='b', type=int, nargs=1,
        help='Index of bootstrap to start with')
    args = parser.parse_args()

    # Varied parameters
    starting_bootstrap = args.starting_bootstrap[0] 

    # Parameters
    seed = 123
    num_cones = 9
    num_epochs = 20
    kfold_index = 0
    num_bootstraps = 100
    num_gpus = 4
    train_size = 1.0

    for bootstrap_index in range(starting_bootstrap, num_bootstraps):

        version = f'numcones_{num_cones}_trainsize_{train_size}_bootstrap_{bootstrap_index}'
        data_dir = os.path.abspath('./src/data')
        experiment_dir = os.path.abspath('./src/bootstrapping/single_iso_pT_efp')
        version_dir = os.path.abspath(f'{experiment_dir}/lightning_logs/{version}')

        # Remove existing checkpoints
        try:
            shutil.rmtree(f'{version_dir}/checkpoints')
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))

        cmd = ['python', f'{experiment_dir}/trainer.py', str(kfold_index),
            str(bootstrap_index), str(train_size), str(num_epochs), str(num_cones), 
            str(num_gpus), str(seed)]
        subprocess.call(cmd)
