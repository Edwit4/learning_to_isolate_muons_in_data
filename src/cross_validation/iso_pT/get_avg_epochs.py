import os
import numpy as np

num_folds = 5
min_cones = 1
max_cones = 18
experiment_dir = './src/cross_validation/iso_pT/lightning_logs'
epochs = np.empty(num_folds)

for i in range(min_cones, max_cones+1):
    for j in range(num_folds):
        version_dir = f'{experiment_dir}/numcones_{i}_trainsize_1.0_fold_{j}'
        ckpt = os.listdir(f'{version_dir}/checkpoints')[0]
        epochs[i] = ckpt.split('=')[1].split('-')[0]

    print(f'num Cones: {i}, epochs: {np.mean(epochs,axis=0)}')