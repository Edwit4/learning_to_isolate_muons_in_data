import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc
from src.utilities.model_evaluation import calc_weak_roc_splot
from src.utilities.models import iso_efp_net
from src.utilities.data_modules import iso_efp_lmdb_data

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('kfold_index', metavar='f', type=int, nargs=1,
        help='Index of kfold')
    parser.add_argument('bootstrap_index', metavar='b', type=int, nargs=1,
        help='Index of bootstrap')
    parser.add_argument('train_size', metavar='s', type=float, nargs=1,
        help='Proportion of total available training set to use')
    parser.add_argument('num_epochs', metavar='e', type=float, nargs=1,
        help='Number of epochs to train for')
    parser.add_argument('num_cones', metavar='c', type=int, nargs=1,
        help='Number of cones to use')
    parser.add_argument('num_gpus', metavar='g', type=int, nargs=1,
        help='Number of gpus to use')
    parser.add_argument('seed', metavar='r', type=int, nargs=1,
        help='Random seed')
    args = parser.parse_args()

    # Varied parameters
    kfold_index = args.kfold_index[0] 
    bootstrap_index = args.bootstrap_index[0] 
    train_size = args.train_size[0]
    num_epochs = args.num_epochs[0]
    num_cones = args.num_cones[0]
    num_gpus = args.num_gpus[0]
    seed = args.seed[0]

    # Parameters
    kfold_index = 0
    M_lo = 70
    M_hi = 110
    pos_lo = 84
    pos_hi = 96
    pT_bound = 25
    eta_bound = 2.1
    min_change = 0.00033 
    num_folds = 5 
    test_size = 0.2 
    effective_batch_size = 1024
    effective_learning_rate = 3e-6
    patience = 10 
    l_size = 128
    test_size = 0.2

    selected_kappa_betas = [[1.0, 1.0], [1.0, 2.0]]
    selected_graphs = [[13, 20], [1, 13]]
    version = f'numcones_{num_cones}_trainsize_{train_size}_bootstrap_{bootstrap_index}'

    num_efps = len(np.concatenate(selected_graphs))

    data_dir = os.path.abspath('./src/data')
    experiment_dir = os.path.abspath('./src/bootstrapping/simsafe_iso_pT_efp')
    version_dir = os.path.abspath(f'{experiment_dir}/lightning_logs/{version}')

    if num_gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = None

    pl.seed_everything(seed)
    data = iso_efp_lmdb_data(data_dir, batch_size=int(effective_batch_size/num_gpus),
        num_folds=num_folds, kfold_index=kfold_index, test_size=test_size,
        train_size=train_size, pT_bound=pT_bound, eta_bound=eta_bound,
        M_lo=M_lo, M_hi=M_hi, pos_lo=pos_lo, pos_hi=pos_hi, num_cones=num_cones,
        kappa_betas=selected_kappa_betas, bootstrap_seed=bootstrap_index+seed,
        efp_indices=selected_graphs, scaler=StandardScaler(), include_sum_pT=True,
        rng=np.random.default_rng(seed=seed))
    model = iso_efp_net(num_cones, num_efps, 
        l_size, effective_learning_rate*num_gpus, version_dir,
        include_sum_pT=True, patience=patience, min_change=min_change,
        batch_size=int(effective_batch_size/num_gpus))
    logger = CSVLogger(save_dir=experiment_dir, version=version, 
        name="lightning_logs")
    trainer = pl.Trainer(gpus=num_gpus, max_epochs=num_epochs, strategy=strategy, 
        logger=logger, limit_val_batches=0, num_sanity_val_steps=0)
    trainer.fit(model, datamodule=data)

    # Test
    ckpts = os.listdir(f'{version_dir}/checkpoints')
    ckpt_epochs = np.array([np.int32(ckpt.split('=')[1].split('-')[0]) for ckpt in ckpts])
    stop_epoch = num_epochs-1
    stop_idx = np.argwhere(ckpt_epochs==stop_epoch).flatten()[0]
    for i in range(len(ckpts)):
        if i != stop_idx:
            try:
                os.remove(f'{version_dir}/checkpoints/{ckpts[i]}')
            except:
                pass
    stop_ckpt_path = f'{version_dir}/checkpoints/{ckpts[stop_idx]}'
    trainer.test(datamodule=data,ckpt_path=stop_ckpt_path)

    test_idx = []
    test_labels = []
    test_outs = []
    test_masses= []
    for i in range(num_gpus):
        test_idx.append(np.load(f'{version_dir}/test_idx_cuda:{i}.npy'))
        test_labels.append(np.load(f'{version_dir}/test_labels_cuda:{i}.npy'))
        test_outs.append(np.load(f'{version_dir}/test_outs_cuda:{i}.npy'))
        test_masses.append(np.load(f'{version_dir}/test_masses_cuda:{i}.npy'))
    test_idx = np.concatenate(test_idx).flatten()
    test_idx, unique_filter = np.unique(test_idx,return_index=True)
    test_labels = np.concatenate(test_labels).flatten()[unique_filter]
    test_outs = np.concatenate(test_outs).flatten()[unique_filter]
    test_masses = np.concatenate(test_masses).flatten()[unique_filter]

    fpr, tpr, thresh = calc_weak_roc_splot(test_masses, test_outs)
    test_auc = auc(fpr,tpr)

    plt.cla()
    plt.plot(fpr, tpr)
    plt.title(f'ROC Curve, AUC = {test_auc:.3f}')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.savefig(f'{version_dir}/test_ROC_curve_full_{test_auc:.3f}.png')
    print('sPlot test AUC: ', test_auc)
