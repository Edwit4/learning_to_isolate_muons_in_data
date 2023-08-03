import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
from src.utilities.data_utils import prep_indices_labels_weights
from src.utilities.datasets import CMS_PFN_lmdb_Dataset, CMS_iso_lmdb_Dataset, \
    sim_PFN_lmdb_Dataset, CMS_iso_efp_lmdb_Dataset

class base_data(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=None, num_folds=None, kfold_index=None, test_size=None,
        train_size=None, pT_bound=None, eta_bound=None, M_lo=None, M_hi=None, pos_lo=None, pos_hi=None,
        scaler=None, rng=None, supervised=None, bootstrap_seed=None):
        super().__init__(self)

        if batch_size is None:
            batch_size = 32
        if num_folds is None:
            num_folds = 5
        if kfold_index is None:
            kfold_index = 0
        if test_size is None:
            test_size = 0.2
        if train_size is None:
            train_size = 1.
        if pT_bound is None:
            pT_bound = 25
        if eta_bound is None:
            eta_bound = 2.1
        if M_lo is None:
            M_lo = 70
        if M_hi is None:
            M_hi = 110
        if pos_lo is None:
            pos_lo = 84
        if pos_hi is None:
            pos_hi = 96
        if rng is None:
            rng = np.random.default_rng(seed=123)
        if supervised is None:
            supervised = False

        self.data_dir = data_dir
        self.num_workers = 4
        self.batch_size = batch_size
        self.num_folds = num_folds
        self.kfold_i = kfold_index
        self.test_size = test_size
        self.train_size = train_size
        self.pT_bound = pT_bound
        self.eta_bound = eta_bound
        self.M_lo = M_lo
        self.M_hi = M_hi
        self.pos_lo = pos_lo 
        self.pos_hi = pos_hi 
        self.scaler = scaler
        self.rng = rng
        self.supervised = supervised
        self.bootstrap_seed = bootstrap_seed

    def setup(self):
        
        self.train_data, self.valid_data, self.test_data = (None, None, None)

        self.train_index, self.valid_index, self.test_index, self.labels, self.weights = \
            prep_indices_labels_weights(self.data_dir, self.num_folds, self.kfold_i,
                self.train_size, self.test_size, self.pT_bound, self.eta_bound, self.M_lo, self.M_hi,
                self.pos_lo, self.pos_hi, self.rng, self.supervised, self.bootstrap_seed)

    def train_dataloader(self):
        
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers,
            pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        
        return DataLoader(self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers,
            pin_memory=True, persistent_workers=True)
  
    def test_dataloader(self):
        
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers,
            pin_memory=True, persistent_workers=True)
    
class iso_lmdb_data(base_data):
    def __init__(self, data_dir, batch_size=None, num_folds=None, kfold_index=None, test_size=None,
        train_size=None, pT_bound=None, eta_bound=None, M_lo=None, M_hi=None, pos_lo=None, pos_hi=None,
        num_cones=None, scaler=None, rng=None, supervised=None, bootstrap_seed=None):
        super().__init__(data_dir, batch_size, num_folds, kfold_index, test_size, train_size, pT_bound, 
            eta_bound, M_lo, M_hi, pos_lo, pos_hi, scaler, rng, supervised, bootstrap_seed)

        if num_cones is None:
            num_cones = -1

        self.num_cones = num_cones
       
    def setup(self, stage: Optional[str] = None):

        super().setup()
        
        # Prep tensor datasets for fitting
        if stage in (None, 'fit'):

            self.train_data = CMS_iso_lmdb_Dataset(self.data_dir,self.train_index,
                self.labels, self.weights, num_cones=self.num_cones, scaler=None)

            if self.scaler is not None:
                scale_dataloader = DataLoader(self.train_data,
                        batch_size=self.batch_size, num_workers=self.num_workers, 
                        pin_memory=True, persistent_workers=True)
                for i, sample in enumerate(scale_dataloader):
                    if i%32==0:
                        print(f'Scaling batch {i}/{len(scale_dataloader)}\r',end='')
                    self.scaler.partial_fit(sample['input'])

            self.train_data.scaler = self.scaler

            self.valid_data = CMS_iso_lmdb_Dataset(self.data_dir,self.valid_index,
                self.labels, self.weights, num_cones=self.num_cones, scaler=self.scaler)

        # Prep tensor dataset for testing
        if stage in (None, 'test'):

            self.test_data = CMS_iso_lmdb_Dataset(self.data_dir,self.test_index,
                self.labels, self.weights, num_cones=self.num_cones, scaler=self.scaler)
                   
class pfn_lmdb_data(base_data):
    def __init__(self, data_dir, batch_size=None, num_folds=None, kfold_index=None, test_size=None,
        train_size=None, pT_bound=None, eta_bound=None, M_lo=None, M_hi=None, pos_lo=None, pos_hi=None,
        scaler=None, rng=None, supervised=None, bootstrap_seed=None):
        super().__init__(data_dir, batch_size, num_folds, kfold_index, test_size, train_size, pT_bound, 
            eta_bound, M_lo, M_hi, pos_lo, pos_hi, scaler, rng, supervised, bootstrap_seed)

    def setup(self, stage: Optional[str] = None):
        
        super().setup()
        
        # Prep tensor datasets for fitting
        if stage in (None, 'fit'):

            self.train_data = CMS_PFN_lmdb_Dataset(self.data_dir, self.train_index,
                self.labels, self.weights, scaler=None)

            if self.scaler is not None:
                scale_dataloader = DataLoader(self.train_data,
                        batch_size=self.batch_size, num_workers=self.num_workers, 
                        pin_memory=True, persistent_workers=True)
                for i, sample in enumerate(scale_dataloader):
                    if i%32==0:
                        print(f'Scaling batch {i}/{len(scale_dataloader)}\r',end='')
                    self.scaler.partial_fit(sample['input'])

            self.train_data.scaler = self.scaler

            self.valid_data = CMS_PFN_lmdb_Dataset(self.data_dir,self.valid_index,
                self.labels, self.weights, scaler=self.scaler)
            
        # Prep tensor dataset for testing
        if stage in (None, 'test'):

            self.test_data = CMS_PFN_lmdb_Dataset(self.data_dir, self.test_index,
                self.labels, self.weights, scaler=self.scaler)

class pfn_sim_lmdb_data(base_data):
    def __init__(self, data_dir, batch_size=None, num_folds=None, kfold_index=None, test_size=None,
        train_size=None, pT_bound=None, eta_bound=None, M_lo=None, M_hi=None, pos_lo=None, pos_hi=None,
        scaler=None, rng=None, supervised=None, bootstrap_seed=None):
        super().__init__(data_dir, batch_size, num_folds, kfold_index, test_size, train_size, pT_bound, 
            eta_bound, M_lo, M_hi, pos_lo, pos_hi, scaler, rng, supervised, bootstrap_seed)

    def setup(self, stage: Optional[str] = None):
        
        super().setup()
        
        # Prep tensor datasets for fitting
        if stage in (None, 'fit'):

            self.train_data = sim_PFN_lmdb_Dataset(self.data_dir, self.train_index,
                self.labels, self.weights, scaler=None)

            if self.scaler is not None:
                scale_dataloader = DataLoader(self.train_data,
                        batch_size=self.batch_size, num_workers=self.num_workers, 
                        pin_memory=True, persistent_workers=True)
                for i, sample in enumerate(scale_dataloader):
                    if i%32==0:
                        print(f'Scaling batch {i}/{len(scale_dataloader)}\r',end='')
                    self.scaler.partial_fit(sample['input'])

            self.train_data.scaler = self.scaler

            self.valid_data = sim_PFN_lmdb_Dataset(self.data_dir,self.valid_index,
                self.labels, self.weights, scaler=self.scaler)
            
        # Prep tensor dataset for testing
        if stage in (None, 'test'):

            self.test_data = sim_PFN_lmdb_Dataset(self.data_dir, self.test_index,
                self.labels, self.weights, scaler=self.scaler)

class iso_efp_lmdb_data(base_data):
    def __init__(self, data_dir, batch_size=None, num_folds=None, kfold_index=None, test_size=None,
        train_size=None, pT_bound=None, eta_bound=None, M_lo=None, M_hi=None, pos_lo=None, pos_hi=None,
        num_cones=None, efp_indices=None, kappa_betas=None, scaler=None, rng=None, supervised=None,
        include_sum_pT=None, bootstrap_seed=None):
        super().__init__(data_dir, batch_size, num_folds, kfold_index, test_size, train_size, pT_bound, 
            eta_bound, M_lo, M_hi, pos_lo, pos_hi, scaler, rng, supervised, bootstrap_seed)

        if num_cones is None:
            num_cones = -1
        if efp_indices is None:
            efp_indices = slice(None)
        if kappa_betas is None:
            kappa_betas = [(1.,1.)]
        if include_sum_pT is None:
            include_sum_pT = True

        self.num_cones = num_cones
        self.efp_indices = efp_indices
        self.kappa_betas = kappa_betas
        self.include_sum_pT = include_sum_pT
       
    def setup(self, stage: Optional[str] = None):

        super().setup()
        
        # Prep tensor datasets for fitting
        if stage in (None, 'fit'):

            self.train_data = CMS_iso_efp_lmdb_Dataset(self.data_dir, self.kappa_betas,
                self.efp_indices, self.train_index, self.labels, self.weights, 
                num_cones=self.num_cones, scaler=None, include_sum_pT=self.include_sum_pT)

            if self.scaler is not None:
                scale_dataloader = DataLoader(self.train_data,
                        batch_size=self.batch_size, num_workers=self.num_workers, 
                        pin_memory=True, persistent_workers=True)
                for i, sample in enumerate(scale_dataloader):
                    if i%32==0:
                        print(f'Scaling batch {i}/{len(scale_dataloader)}\r',end='')
                    self.scaler.partial_fit(sample['input'])

            self.train_data.scaler = self.scaler

            self.valid_data = CMS_iso_efp_lmdb_Dataset(self.data_dir, self.kappa_betas,
                self.efp_indices, self.valid_index, self.labels, self.weights, 
                num_cones=self.num_cones, scaler=self.scaler, include_sum_pT=self.include_sum_pT)

        # Prep tensor dataset for testing
        if stage in (None, 'test'):

            self.test_data = CMS_iso_efp_lmdb_Dataset(self.data_dir, self.kappa_betas,
                self.efp_indices, self.test_index, self.labels, self.weights, 
                num_cones=self.num_cones, scaler=self.scaler, include_sum_pT=self.include_sum_pT)