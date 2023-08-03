import os
import shutil
import argparse
import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from src.utilities.models import pfn
from src.utilities.data_modules import pfn_lmdb_data
from src.utilities.scalers import pfn_standard_scaler
from src.utilities.model_evaluation import calc_weak_roc_splot

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('kfold_index', metavar='f', type=int, nargs=1,
        help='Index of cross validation fold')
    parser.add_argument('train_size', metavar='s', type=float, nargs=1,
        help='Proportion of total available training set to use')
    parser.add_argument('num_gpus', metavar='g', type=int, nargs=1,
        help='Number of gpus to use')
    args = parser.parse_args()

    # Varied parameters
    kfold_index = args.kfold_index[0] # Index of fold to evaluate
    train_size = args.train_size[0]
    num_gpus = args.num_gpus[0]

    # Parameters
    num_folds = 5 # Total num kfolds
    seed = 123
    M_lo = 70
    M_hi = 110
    pos_lo = 84
    pos_hi = 96
    pT_bound = 25
    eta_bound = 2.1
    test_size = 0.2 # Proportion of full dataset to use for testing
    num_epochs = 300
    effective_batch_size = 256
    effective_learning_rate = 0.5e-6
    patience = 10
    min_change = 0.00033 # Minimum change for early stopping
    scaler = pfn_standard_scaler()
    Phi_size = 256
    F_size = 256
    l_size = 512
    test_size = 0.2

    data_dir = os.path.abspath('src/data')
    version = f'trainsize_{train_size}_fold_{kfold_index}'
    experiment_dir = 'src/training/pfn'
    version_dir = f'{experiment_dir}/lightning_logs/{version}'

    if num_gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = None

    # Remove existing checkpoints
    try:
        shutil.rmtree(f'{version_dir}/checkpoints')
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))

    # Remove previously stored test objects
    if os.path.exists(version_dir):
        for f in os.listdir(version_dir):
            if f.endswith('.npy'):
                try:
                    os.remove(f'{version_dir}/{f}')
                except:
                    pass

    pl.seed_everything(seed)
    data = pfn_lmdb_data(data_dir, batch_size=int(effective_batch_size/num_gpus), num_folds=num_folds, kfold_index=kfold_index,
                test_size=test_size, train_size=train_size, pT_bound=pT_bound, eta_bound=eta_bound, 
                M_lo=M_lo, M_hi=M_hi, pos_lo=pos_lo, pos_hi=pos_hi, scaler=scaler,
                rng=np.random.default_rng(seed=seed))
    model = pfn(Phi_size, F_size, l_size, effective_learning_rate*num_gpus, version_dir)
    logger = CSVLogger(save_dir=experiment_dir, version=version, name="lightning_logs")
    model_checkpoint_callback = ModelCheckpoint(monitor='val_loss', verbose=False, mode='min', 
        save_top_k=-1)
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=min_change, patience=patience,
        verbose=False, mode='min')
    callbacks = [early_stop_callback, model_checkpoint_callback]
    trainer = pl.Trainer(gpus=num_gpus, max_epochs=num_epochs, strategy=strategy, 
        callbacks=callbacks, logger=logger, num_sanity_val_steps=0)
    trainer.fit(model, datamodule=data)

    # Test
    ckpts = os.listdir(f'{version_dir}/checkpoints')
    ckpt_epochs = np.array([np.int32(ckpt.split('=')[1].split('-')[0]) for ckpt in ckpts])
    stopped_epoch = early_stop_callback.state_dict()['stopped_epoch']
    wait_epochs = early_stop_callback.state_dict()['wait_count']
    if stopped_epoch > patience:
        early_stop_epoch = stopped_epoch - wait_epochs
    else:
        early_stop_epoch = stopped_epoch
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