import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.model_selection import StratifiedKFold

def make_weights_kde(sample, target, bw_factor='scott'):

    sample = sample.squeeze()
    target = target.squeeze()

    sample_kernel = gaussian_kde(sample)
    target_kernel = gaussian_kde(target,bw_method=bw_factor)
    sample_kernel.set_bandwidth(target_kernel.factor)
    weights = target_kernel(sample) / sample_kernel(sample)

    return weights

def transform_data(self, transformers, train_inputs, valid_inputs, test_inputs):
    for t in transformers:
        t.fit(train_inputs)
        train_inputs = t.transform(train_inputs)
        valid_inputs = t.transform(valid_inputs)
        test_inputs = t.transform(test_inputs)
    return train_inputs, valid_inputs, test_inputs

def prep_indices_labels_weights(data_dir, num_folds, kfold_index, train_size,
    test_size, pT_bound, eta_bound, M_lo, M_hi, pos_lo, pos_hi, rng, 
    supervised=False, bootstrap_seed=None):

    if supervised:
        m = pd.read_csv(f'{data_dir}/sim_csvs/masses.csv', header=None).to_numpy().squeeze()
        pT = pd.read_csv(f'{data_dir}/sim_csvs/muon_pTs.csv', header=None).to_numpy()
        eta = pd.read_csv(f'{data_dir}/sim_csvs/muon_etas.csv', header=None).to_numpy()
        phi = pd.read_csv(f'{data_dir}/sim_csvs/muon_phis.csv', header=None).to_numpy()
        label = pd.read_csv(f'{data_dir}/sim_csvs/muon_labels.csv', header=None).to_numpy()

        selection_1 = (m < 96) & (m > 84)
        selection_2 = np.invert(selection_1)
        label[selection_1] = 1
        label[selection_2] = 0
        selection = np.arange(len(label))
        rng.shuffle(selection)

        pos_flag = (label==1)
    else:
        m = pd.read_csv(f'{data_dir}/cms_csvs/dimuon_invariant_mass.csv', header=None).to_numpy().squeeze()
        pT = pd.read_csv(f'{data_dir}/cms_csvs/muon_pTs.csv', header=None).to_numpy()
        eta = pd.read_csv(f'{data_dir}/cms_csvs/muon_etas.csv', header=None).to_numpy()
        phi = pd.read_csv(f'{data_dir}/cms_csvs/muon_phis.csv', header=None).to_numpy()
        q = pd.read_csv(f'{data_dir}/cms_csvs/muon_charges.csv', header=None).to_numpy()

        M_selection = ((m > M_lo) & (m < M_hi))
        eta_selection = (np.abs(eta)[:,0] < eta_bound) & (np.abs(eta)[:,1] < eta_bound) 
        pT_selection = (pT[:,0] > pT_bound) & (pT[:,1] > pT_bound) 
        selection = M_selection & eta_selection & pT_selection
        selection = np.argwhere(selection).flatten()

        label = np.ones(len(m))
        label[m < pos_lo] = 0
        label[m > pos_hi] = 0
        label[np.prod(q,axis=1)>0] = 0

        #pos_flag = (label==1)
        pos_flag = (np.prod(q,axis=1) < 0)

        shuffle = np.arange(len(selection))
        rng.shuffle(shuffle)
        selection = selection[shuffle]

    kf = StratifiedKFold(n_splits=num_folds+1, shuffle=False)
    split = list(kf.split(selection, y=label[selection]))
    test_split = split[0][1]
    train_split, valid_split = split[kfold_index+1]
    train_split = np.delete(train_split, 
        np.argwhere(np.isin(train_split,test_split)).flatten())
    valid_split = np.delete(valid_split, 
        np.argwhere(np.isin(valid_split,test_split)).flatten())
    train_index = selection[train_split] 
    valid_index = selection[valid_split]
    test_index = selection[test_split]

    if bootstrap_seed is not None:
        bootstrap_rng = np.random.default_rng(bootstrap_seed)
        bootstrap_train_index = np.concatenate([train_index,valid_index])
        bootstrap_rng.shuffle(bootstrap_train_index)
        train_pos_i = np.argwhere(label[bootstrap_train_index]==1).squeeze()
        train_neg_i = np.argwhere(label[bootstrap_train_index]==0).squeeze()
        train_pos = bootstrap_rng.choice(bootstrap_train_index[train_pos_i],
            size=len(train_pos_i), replace=True, shuffle=False)
        train_neg = bootstrap_rng.choice(bootstrap_train_index[train_neg_i],
            size=len(train_neg_i), replace=True, shuffle=False)
        train_index = np.concatenate([train_pos,train_neg])
        bootstrap_rng.shuffle(train_index)

    w = np.ones(label.shape)
    pos = np.argwhere(pos_flag[train_index].squeeze()).squeeze()
    neg = np.argwhere(np.invert(pos_flag[train_index].squeeze())).squeeze()
    w_train = np.ones(label[train_index].shape)
    w_train[neg] = make_weights_kde(np.array([eta[train_index][neg][:,0], pT[train_index][neg][:,0]]),
                    np.array([eta[train_index][pos][:,0], pT[train_index][pos][:,0]])).reshape(
                        w_train[neg].shape)
    pos = np.argwhere(pos_flag[valid_index].squeeze()).squeeze()
    neg = np.argwhere(np.invert(pos_flag[valid_index].squeeze())).squeeze()
    w_valid = np.ones(label[valid_index].shape)
    w_valid[neg] = make_weights_kde(np.array([eta[valid_index][neg][:,0], pT[valid_index][neg][:,0]]),
                    np.array([eta[valid_index][pos][:,0], pT[valid_index][pos][:,0]])).reshape(
                        w_valid[neg].shape)

    unique, unique_counts = np.unique(label[train_index],return_counts=True)
    min_index = np.argmin(unique_counts)
    min_label = unique[min_index]
    min_weight = unique_counts[np.int32(np.abs(min_index-1))] / unique_counts[min_index]
    w_train[label[train_index]==min_label] = w_train[label[train_index]==min_label]*min_weight
    w_train = w_train / np.sum(w_train)

    unique, unique_counts = np.unique(label[valid_index],return_counts=True)
    min_index = np.argmin(unique_counts)
    min_label = unique[min_index]
    min_weight = unique_counts[np.int32(np.abs(min_index-1))] / unique_counts[min_index]
    w_valid[label[valid_index]==min_label] = w_valid[label[valid_index]==min_label]*min_weight
    w_valid = w_valid / np.sum(w_valid)
    
    w[train_index] = w_train
    w[valid_index] = w_valid

    return np.sort(train_index), np.sort(valid_index), np.sort(test_index), label, w

def atleast_2d(*arys):
    # Taken from numpy code
    res = []
    for ary in arys:
        if ary.ndim == 0:
            result = ary.reshape(1, 1)
        elif ary.ndim == 1:
            result = ary[None, :]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res

