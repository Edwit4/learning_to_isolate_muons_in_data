import os
import shutil
import argparse
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import auc, roc_curve
from src.utilities.model_evaluation import calc_weak_roc_splot
from src.utilities.plotting import plot_loss_history, plot_outputs, plot_roc_curve
from src.utilities.models import iso_efp_net
from src.utilities.data_modules import iso_efp_lmdb_data

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('kfold_index', metavar='f', type=int, nargs=1,
        help='Index of cross validation fold')
    parser.add_argument('train_size', metavar='s', type=float, nargs=1,
        help='Proportion of total available training set to use')
    parser.add_argument('num_cones', metavar='c', type=int, nargs=1,
        help='Number of cones to use')
    parser.add_argument('kappa', metavar='k', type=float, nargs=1,
        help='EFP kappa parameter')
    parser.add_argument('beta', metavar='b', type=float, nargs=1,
        help='EFP beta parameter')
    parser.add_argument('graph_index', metavar='i', type=int, nargs=1,
        help='EFP graph index')
    parser.add_argument('num_gpus', metavar='g', type=int, nargs=1,
        help='Number of gpus to use')
    args = parser.parse_args()

    # Varied parameters
    kfold_index = args.kfold_index[0] # Index of fold to evaluate
    train_size = args.train_size[0]
    num_cones = args.num_cones[0]
    kappa = args.kappa[0]
    beta = args.beta[0]
    graph_index = args.graph_index[0]
    num_gpus = args.num_gpus[0]

    # Parameters
    seed = 123
    M_lo = 70
    M_hi = 110
    pos_lo = 84
    pos_hi = 96
    pT_bound = 25
    eta_bound = 2.1
    min_change = 0.000333 # Minimum change for early stopping
    num_folds = 5 # Total num kfolds
    test_size = 0.2 # Proportion of full dataset to use for testing
    effective_batch_size = 256
    num_epochs = 300
    effective_learning_rate = 1e-6 
    patience = 10 
    l_size = 256
    test_size = 0.2

    version = f'kappa_{kappa}_beta_{beta}_numcones_{num_cones}_trainsize_{train_size}_fold_{kfold_index}'
    selected_kappa_betas = [[kappa, beta]]
    selected_graphs = [[graph_index]]

    if len(selected_graphs) > 0:
        num_efps = len(np.concatenate(selected_graphs))
    else:
        num_efps = 0

    if num_gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = None

    data_dir = os.path.abspath('./src/data')
    experiment_dir = './src/cross_validation/iso_efp_quadratic'
    version_dir = os.path.abspath(f'{experiment_dir}/lightning_logs/{version}')

    # Remove existing checkpoints
    try:
        shutil.rmtree(f'{version_dir}/checkpoints')
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))

    pl.seed_everything(seed)
    data = iso_efp_lmdb_data(data_dir, batch_size=int(effective_batch_size/num_gpus),
        num_folds=num_folds, kfold_index=kfold_index, test_size=test_size,
        train_size=train_size, pT_bound=pT_bound, eta_bound=eta_bound,
        M_lo=M_lo, M_hi=M_hi, pos_lo=pos_lo, pos_hi=pos_hi, num_cones=num_cones,
        kappa_betas=selected_kappa_betas,
        efp_indices=selected_graphs, scaler=StandardScaler(), include_sum_pT=True,
        rng=np.random.default_rng(seed=seed))
    model = iso_efp_net(num_cones, num_efps, 
        l_size, effective_learning_rate*num_gpus, version_dir,
        include_sum_pT=True, batch_size=int(effective_batch_size/num_gpus))
    logger = CSVLogger(save_dir=experiment_dir, version=version, name="lightning_logs")
    model_checkpoint_callback = ModelCheckpoint(monitor='val_loss', verbose=False, mode='min', 
        save_top_k=-1)
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=min_change, patience=patience,
        verbose=False, mode='min')
    callbacks = [early_stop_callback, model_checkpoint_callback]
    trainer = pl.Trainer(gpus=num_gpus, max_epochs=num_epochs, strategy=strategy, 
        callbacks=callbacks, logger=logger)
    trainer.fit(model, datamodule=data)

    # Test
    ckpts = os.listdir(f'{version_dir}/checkpoints')
    ckpt_epochs = np.array([np.int32(ckpt.split('=')[1].split('-')[0]) for ckpt in ckpts])
    stopped_epoch = early_stop_callback.state_dict()['stopped_epoch']
    wait_epochs = early_stop_callback.state_dict()['wait_count']
    early_stop_epoch = stopped_epoch - wait_epochs
    early_stop_idx = np.argwhere(ckpt_epochs==early_stop_epoch).flatten()[0]
    for i in range(len(ckpts)):
        if i != early_stop_idx:
            try:
                os.remove(f'{version_dir}/checkpoints/{ckpts[i]}')
            except:
                pass
    early_stop_ckpt_path = f'{version_dir}/checkpoints/{ckpts[early_stop_idx]}'
    trainer.test(datamodule=data,ckpt_path=early_stop_ckpt_path)

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