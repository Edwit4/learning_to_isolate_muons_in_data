import numpy as np
from sklearn.metrics import auc
from src.utilities.model_evaluation import calc_weak_roc_splot


experiment_dir = './src/cross_validation/iso_efp_quadratic'
num_gpus = 4
num_folds = 5
kappas = [-1.,0.,0.25,0.5,1.,2.]
betas = [0.25,0.5,1.,2.,3.,4.]
kbs = [[k,b] for k in kappas for b in betas]

aucs = np.zeros((len(kappas)*len(betas),num_folds))
for i, (kappa, beta) in enumerate(kbs):
    for j in range(num_folds):

        print(f'kb {i+1}/{len(kbs)}, fold {j+1}/{num_folds}\r',end='')
        version_dir = f'{experiment_dir}/lightning_logs/kappa_{kappa}_beta_{beta}_numcones_9_trainsize_1.0_fold_{j}'

        test_idx = []
        test_labels = []
        test_outs = []
        test_masses= []
        for k in range(num_gpus):
            test_idx.append(np.load(f'{version_dir}/test_idx_cuda:{k}.npy'))
            test_labels.append(np.load(f'{version_dir}/test_labels_cuda:{k}.npy'))
            test_outs.append(np.load(f'{version_dir}/test_outs_cuda:{k}.npy'))
            test_masses.append(np.load(f'{version_dir}/test_masses_cuda:{k}.npy'))
        test_idx = np.concatenate(test_idx).flatten()
        test_idx, unique_filter = np.unique(test_idx,return_index=True)
        test_labels = np.concatenate(test_labels).flatten()[unique_filter]
        test_outs = np.concatenate(test_outs).flatten()[unique_filter]
        test_masses = np.concatenate(test_masses).flatten()[unique_filter]

        fpr, tpr, thresh = calc_weak_roc_splot(test_masses, test_outs)
        aucs[i][j] = auc(fpr,tpr)

aucs = np.array(aucs)
avg_aucs = np.mean(aucs,axis=1)
std_aucs = np.std(aucs,axis=1)
max_auc_i = np.argmax(avg_aucs)
print(f'Best AUC: {avg_aucs[max_auc_i]} +- {std_aucs[max_auc_i]}, with kappa {kbs[max_auc_i][0]}, beta {kbs[max_auc_i][1]}')