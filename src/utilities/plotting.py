import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_from_log(keys, version_dir):
    metrics = [None for _ in keys]

    try:
        log = pd.read_csv(f'{version_dir}/metrics.csv')
    except:
        return metrics

    for i, key in enumerate(keys):
        if key in log.columns:
            metrics[i] = log[key][log[key].notna()].to_numpy()

    return metrics

def plot_outputs(model, version_dir):
    test_labels = model.test_labels
    test_outs = model.test_outs
    plt.cla()
    _, bins, _ = plt.hist(test_outs[test_labels==1], label='Signal', alpha=0.5)#, density=True)
    plt.hist(test_outs[test_labels==0], bins=bins, label='Background', alpha=0.5)#, density=True)
    plt.title('Test Outputs')
    plt.xlabel('Output')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'{version_dir}/test_outputs.png')

def plot_roc_curve(model, version_dir):
    plt.cla()
    plt.plot(model.fpr, model.tpr)
    plt.title(f'ROC Curve, AUC = {model.auc:.3f}')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.savefig(f'{version_dir}/test_ROC_curve_full.png')

def plot_loss_history(version_dir, patience=None):
    train_loss, valid_loss = load_from_log(('train_loss', 'val_loss'), version_dir)
    
    if (train_loss is not None) and (valid_loss is not None):

        plt.cla()
        plt.plot(np.arange(len(train_loss))-0.5, train_loss, label='train')
        plt.plot(np.arange(len(valid_loss)), valid_loss, label='valid')
        if patience is not None:
            plt.axvline(x=[0,len(train_loss)-patience][int(len(train_loss)>patience)], color='r', linestyle='dashed', label='early stopping')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{version_dir}/loss_history.png')

    elif (train_loss is not None) and (valid_loss is None):

        plt.cla()
        plt.plot(np.arange(len(train_loss))-0.5, train_loss, label='train')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{version_dir}/loss_history.png')

def plot_auc_history(version_dir, patience=None):

    valid_auc_history = load_from_log(('val_auc',), version_dir)[0]

    if valid_auc_history is not None:

        plt.cla()
        plt.plot(np.arange(len(valid_auc_history)), valid_auc_history)
        plt.title('Valid AUC History')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.savefig(f'{version_dir}/valid_auc_history.png')
