import os
import lmdb
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from src.utilities.data_utils import atleast_2d

class CMS_iso_lmdb_Dataset(Dataset):
    def __init__(self, data_dir, selected_idx, labels, weights, num_cones=-1, scaler=None):
        self.data_dir = data_dir
        self.selected_idx = selected_idx
        self.m = pd.read_csv(f'{data_dir}/cms_csvs/dimuon_invariant_mass.csv', header=None).to_numpy().squeeze()
        self.q = pd.read_csv(f'{data_dir}/cms_csvs/muon_charges.csv', header=None).to_numpy()
        self.label = labels
        self.weight = weights
        self.scaler = scaler
        self.env = None
        self.txn = None
        self.num_cones = num_cones

    def _init_db(self):
        self.env = lmdb.open(f'{self.data_dir}/iso', subdir=os.path.isdir(f'{self.data_dir}/iso'),
            readonly=True, lock=False,
            readahead=False, meminit=False)
        self.txn = self.env.begin()
        
    def __getitem__(self, index):
        idx = self.selected_idx[index]
        mass = self.m[idx]
        charge = self.q[idx]
        label = self.label[idx]
        weight = self.weight[idx]
       
        if self.env is None:
            self._init_db()
        lmdb_data = np.frombuffer(self.txn.get(str(idx).encode()))
        selected_lmdb_data = lmdb_data[-self.num_cones:]

        if self.scaler is not None:
            scaled_lmdb_data = self.scaler.transform(atleast_2d(selected_lmdb_data))
            input_data = torch.tensor(scaled_lmdb_data)
        else:
            input_data = torch.tensor(selected_lmdb_data)

        sample = {'mass': mass, 'charge': charge, 'label': label, 'weight': weight, 
            'input': input_data, 'index': idx}

        return sample
 
    def __len__(self):
        return len(self.label[self.selected_idx])

class CMS_PFN_lmdb_Dataset(Dataset):
    def __init__(self, data_dir, selected_idx, labels, weights, scaler=None):
        self.data_dir = data_dir
        self.selected_idx = selected_idx
        self.m = pd.read_csv(f'{data_dir}/cms_csvs/dimuon_invariant_mass.csv', header=None).to_numpy().squeeze()
        self.q = pd.read_csv(f'{data_dir}/cms_csvs/muon_charges.csv', header=None).to_numpy()
        self.pT = pd.read_csv(f'{data_dir}/cms_csvs/muon_pTs.csv', header=None).to_numpy()[:,0]
        self.label = labels
        self.weight = weights
        self.scaler = scaler
        self.env = None
        self.txn = None

    def _init_db(self):
        self.env = lmdb.open(f'{self.data_dir}/pfn', subdir=os.path.isdir(f'{self.data_dir}/pfn'),
            readonly=True, lock=False,
            readahead=False, meminit=False)
        self.txn = self.env.begin(buffers=True)
        
    def __getitem__(self, index):
        idx = self.selected_idx[index]
        mass = self.m[idx]
        charge = self.q[idx]
        muon_pT = self.pT[idx]
        label = self.label[idx]
        weight = self.weight[idx]

        if self.env is None:
            self._init_db()
        lmdb_data = self.txn.get(str(idx).encode())
        lmdb_data = np.frombuffer(lmdb_data).reshape(128,4)
        pT_scaled_data = np.empty(lmdb_data.shape)
        pT_scaled_data[:,0] = lmdb_data[:,0] / muon_pT
        pT_scaled_data[:,1:] = lmdb_data[:,1:]
        pT_scaled_data = torch.tensor(pT_scaled_data)

        if self.scaler is not None:
            pT_scaled_data = self.scaler.transform(pT_scaled_data)
        sample = {'mass': mass, 'charge': charge, 'label': label, 'weight': weight, 
            'input': pT_scaled_data, 'index': idx}
        
        return sample
 
    def __len__(self):
        return len(self.label[self.selected_idx])

class sim_PFN_lmdb_Dataset(Dataset):
    def __init__(self, data_dir, selected_idx, labels, weights, scaler=None):
        self.data_dir = data_dir
        self.selected_idx = selected_idx
        self.pT = pd.read_csv(f'{data_dir}/sim_csvs/muon_pTs.csv', header=None).to_numpy()
        self.m = pd.read_csv(f'{data_dir}/sim_csvs/masses.csv', header=None).to_numpy()
        self.true_label = pd.read_csv(f'{data_dir}/sim_csvs/muon_labels.csv', header=None).to_numpy()
        self.label = labels
        self.weight = weights
        self.scaler = scaler
        self.env = None
        self.txn = None

    def _init_db(self):
        self.env = lmdb.open(f'{self.data_dir}/pfn_sim_fixed', subdir=os.path.isdir(f'{self.data_dir}/pfn_sim_fixed'),
            readonly=True, lock=False,
            readahead=False, meminit=False)
        self.txn = self.env.begin()
        
    def __getitem__(self, index):
        idx = self.selected_idx[index]
        mass = self.m[idx]
        muon_pT = self.pT[idx]
        label = self.label[idx][0]
        true_label = self.true_label[idx]
        weight = self.weight[idx]

        if self.env is None:
            self._init_db()
        lmdb_data = self.txn.get(str(idx).encode())
        lmdb_data = np.frombuffer(lmdb_data).reshape(512,3)
        #sort = np.argsort(lmdb_data[:,0])[::-1]
        #lmdb_data = lmdb_data[sort][:128,:]
        pT_scaled_data = np.empty(lmdb_data.shape)
        pT_scaled_data[:,0] = lmdb_data[:,0] / muon_pT
        pT_scaled_data[:,1:] = lmdb_data[:,1:]
        in_rad = np.sqrt(pT_scaled_data[:,1]**2 + pT_scaled_data[:,2]**2)<=0.4
        pT_scaled_data[:,0][np.invert(in_rad)] = 0
        pT_scaled_data[:,1][np.invert(in_rad)] = 0
        pT_scaled_data[:,2][np.invert(in_rad)] = 0
        pT_scaled_data = torch.tensor(pT_scaled_data)

        if self.scaler is not None:
            pT_scaled_data = self.scaler.transform(pT_scaled_data)
        # Just using 'label' as placeholder here since mass and charge aren't present / used, fix this later
        sample = {'mass': mass, 'charge': np.nan, 'label': label, 'weight': weight, 
            'input': pT_scaled_data, 'index': idx, 'true_label': true_label}
        
        return sample
 
    def __len__(self):
        return len(self.label[self.selected_idx])

class CMS_iso_efp_lmdb_Dataset(Dataset):
    def __init__(self, data_dir, kappa_betas, efp_indices, selected_idx, labels, weights, num_cones=-1,
        scaler=None, include_sum_pT=True):
        self.data_dir = data_dir
        self.selected_idx = selected_idx
        self.m = pd.read_csv(f'{self.data_dir}/cms_csvs/dimuon_invariant_mass.csv',
            header=None).to_numpy().squeeze()
        self.q = pd.read_csv(f'{self.data_dir}/cms_csvs/muon_charges.csv', header=None).to_numpy()
        self.label = labels
        self.weight = weights
        self.scaler = scaler
        self.num_cones = num_cones
        self.include_sum_pT = include_sum_pT
        self.pT = pd.read_csv(f'{data_dir}/cms_csvs/muon_pTs.csv', header=None).to_numpy()[:,0]
        self.kappa_betas = kappa_betas
        self.efp_indices = efp_indices
        self.efp_envs = [None for kb in kappa_betas]
        self.efp_txns = [None for kb in kappa_betas]
        self.pfn_env = None
        self.pfn_txn = None
        self.iso_env = None
        self.iso_txn = None

    def _init_db(self):
        self.pfn_env = lmdb.open(f'{self.data_dir}/pfn', subdir=os.path.isdir(f'{self.data_dir}/pfn'),
            readonly=True, lock=False,
            readahead=False, meminit=False)
        self.pfn_txn = self.pfn_env.begin(buffers=True)
        self.iso_env = lmdb.open(f'{self.data_dir}/iso', subdir=os.path.isdir(f'{self.data_dir}/iso'),
            readonly=True, lock=False,
            readahead=False, meminit=False)
        self.iso_txn = self.iso_env.begin(buffers=True)
        self.efp_envs = [lmdb.open(f'{self.data_dir}/cms_efp_k{k}_b{b}', 
            subdir=os.path.isdir(f'{self.data_dir}/cms_efp_k{k}_b{b}'),
            readonly=True, lock=False,
            readahead=False, meminit=False) for k,b in self.kappa_betas]
        self.efp_txns = [env.begin(buffers=True) for env in self.efp_envs]
        
    def __getitem__(self, index):
        idx = self.selected_idx[index]
        mass = self.m[idx]
        charge = self.q[idx]
        label = self.label[idx]
        weight = self.weight[idx]

        initialized = True
        for env in self.efp_envs:
            if env is None:
                initialized = False
        if self.pfn_env is None:
            initialized = False
        if self.iso_env is None:
            initialized = False
        if not initialized:
            self._init_db()

        if self.include_sum_pT:
            pfn_data = self.pfn_txn.get(str(idx).encode())
            pfn_data = np.frombuffer(pfn_data).reshape(128,4)
            sum_pT = np.sum(pfn_data[:,0])
        if len(self.kappa_betas) > 0:
            efp_data = np.concatenate([np.frombuffer(txn.get(str(idx).encode()))[i]
                for i,txn in zip(self.efp_indices,self.efp_txns)])
        else:
            efp_data = []
        iso_data = np.frombuffer(self.iso_txn.get(str(idx).encode()))
        selected_iso_data = iso_data[-self.num_cones:]

        if self.include_sum_pT:
            iso_efp_data = np.concatenate([selected_iso_data,efp_data,[sum_pT]])
        else:
            iso_efp_data = np.concatenate([selected_iso_data,efp_data])

        if self.scaler is not None:
            scaled_iso_efp_data = self.scaler.transform(atleast_2d(iso_efp_data))
            input_data = torch.tensor(scaled_iso_efp_data)
        else:
            input_data = torch.tensor(iso_efp_data)

        sample = {'mass': mass, 'charge': charge, 'label': label, 'weight': weight, 
            'input': input_data, 'index': idx}
        
        return sample
 
    def __len__(self):
        return len(self.label[self.selected_idx])