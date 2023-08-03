import numpy as np
from sklearn.metrics import auc
from src.utilities.model_evaluation import calc_weak_roc_splot

num_bootstraps = 45
experiment_dir = 'src/bootstrapping/single_iso_pT_efp/lightning_logs'
num_gpus = 4

aucs = []
for i in range(num_bootstraps):
    print('\r'*100,end='')
    print(f'Processing bootstrap {i}/{num_bootstraps}\r',end='')
    bootstrap_dir = f'{experiment_dir}/numcones_9_trainsize_1.0_bootstrap_{i}'

    test_idx = []
    test_outs = []
    test_masses= []
    for i in range(num_gpus):
        test_idx.append(np.load(f'{bootstrap_dir}/test_idx_cuda:{i}.npy'))
        test_outs.append(np.load(f'{bootstrap_dir}/test_outs_cuda:{i}.npy'))
        test_masses.append(np.load(f'{bootstrap_dir}/test_masses_cuda:{i}.npy'))
    test_idx = np.concatenate(test_idx).flatten()
    test_idx, unique_filter = np.unique(test_idx,return_index=True)
    test_outs = np.concatenate(test_outs).flatten()[unique_filter]
    test_masses = np.concatenate(test_masses).flatten()[unique_filter]

    fpr, tpr, thresh = calc_weak_roc_splot(test_masses, test_outs)
    aucs.append(auc(fpr,tpr))

print(aucs)
auc_avg = np.mean(aucs)
auc_std = np.std(aucs, ddof=1)
auc_se = auc_std/np.sqrt(num_bootstraps)*1.96
print('')
print(f'Average AUC: {auc_avg}, Error: {auc_se} with 95% confidence')