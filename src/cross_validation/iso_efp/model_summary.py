import os
import numpy as np
from sklearn.metrics import auc
from src.utilities.model_evaluation import calc_weak_roc_splot

experiment_dir = 'src/cross_validation/iso_efp/lightning_logs'
num_folds = 5

aucs = np.empty(num_folds)
for i in range(num_folds):
    version_dir = f'{experiment_dir}/numcones_9_trainsize_1.0_fold_{i}'
    test_idx = []
    test_labels = []
    test_outs = []
    test_masses= []
    gpus = np.unique([f.split(':')[1][0] 
        for f in os.listdir(version_dir) if ':' in f]).astype(np.int32)
    for j in gpus:
        test_idx.append(np.load(f'{version_dir}/test_idx_cuda:{j}.npy'))
        test_labels.append(np.load(f'{version_dir}/test_labels_cuda:{j}.npy'))
        test_outs.append(np.load(f'{version_dir}/test_outs_cuda:{j}.npy'))
        test_masses.append(np.load(f'{version_dir}/test_masses_cuda:{j}.npy'))
    test_idx = np.concatenate(test_idx).flatten()
    test_idx, unique_filter = np.unique(test_idx,return_index=True)
    test_labels = np.concatenate(test_labels).flatten()[unique_filter]
    test_outs = np.concatenate(test_outs).flatten()[unique_filter]
    test_masses = np.concatenate(test_masses).flatten()[unique_filter]

    fpr, tpr, thresh = calc_weak_roc_splot(test_masses, test_outs)
    aucs[i] = auc(fpr,tpr)

auc_avg = np.mean(aucs)
auc_std = np.std(aucs)
print(aucs)
print(f'{auc_avg} +- {auc_std}')