import os
import numpy as np

num_folds = 5
num_cones = 9
experiment_dir = './src/cross_validation/iso_efp_quadratic/lightning_logs'
epochs = np.empty(num_folds)

for i in range(num_folds):
    version_dir = f'{experiment_dir}/kappa_1.0_beta_0.25_numcones_9_trainsize_1.0_fold_{i}'
    ckpt = os.listdir(f'{version_dir}/checkpoints')[0]
    epochs[i] = ckpt.split('=')[1].split('-')[0]

print(epochs)
print(np.mean(epochs,axis=0))