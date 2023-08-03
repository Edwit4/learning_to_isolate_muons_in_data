import os
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
from sklearn.metrics import auc, roc_curve
from src.utilities.model_evaluation import calc_weak_roc_splot
from src.utilities.plotting import plot_loss_history, plot_auc_history

class base_classifier_net(pl.LightningModule):
    def __init__(self, learning_rate, version_dir, supervised=False,
        patience=None, min_change=None, batch_size=None):
        super().__init__()
        self.version_dir = version_dir
        self.learning_rate = learning_rate
        self.supervised = supervised
        self.patience = patience
        self.min_change = min_change 
        self.batch_size = batch_size
        self.criterion = nn.BCELoss(reduction='none')

        self.save_hyperparameters('learning_rate')
        self.save_hyperparameters('batch_size')
        self.save_hyperparameters('patience')
        self.save_hyperparameters('min_change')

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_fit_start(self):
        try:
            os.makedirs(self.version_dir)
        except:
            pass

    def training_step(self, train_batch, batch_idx):
        sample = train_batch
        labels = sample['label'].float()
        w = sample['weight'].float()
        outputs = self.forward(sample).flatten()
        loss = self.criterion(outputs, labels)
        loss = w * loss
        loss = loss.sum() / w.sum()
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def training_epoch_end(self, dict_list):

        plot_loss_history(self.version_dir)

    def validation_step(self, val_batch, batch_idx):
        sample = val_batch 
        labels = sample['label'].float()
        w = sample['weight'].float()
        masses = sample['mass'].float()
        outputs = self.forward(sample).flatten()
        loss = self.criterion(outputs, labels)
        loss = w * loss
        loss = loss.sum() / w.sum()
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        return {'outputs': outputs, 'masses': masses, 'labels': labels}

    def validation_epoch_end(self, dict_list):
        full_dict = {k: [] for k in dict_list[0].keys()}
        for d in dict_list:
            for k in full_dict.keys():
                full_dict[k].append(d[k])
        for k in full_dict.keys():
            full_dict[k] = torch.cat(full_dict[k]).cpu().detach().numpy().squeeze()

        plot_loss_history(self.version_dir)
            
        if len(full_dict['outputs']) > 0:

            if not self.supervised:
                fpr, tpr, thresh = calc_weak_roc_splot(full_dict['masses'], full_dict['outputs'])
            else:
                fpr, tpr, thresh = roc_curve(full_dict['labels'],full_dict['outputs'])
            valid_auc = auc(fpr,tpr)
            self.log('val_auc', valid_auc, sync_dist=True)

            plot_auc_history(self.version_dir)

    def test_step(self, test_batch, batch_idx):

        sample = test_batch 

        labels = sample['label'].float()
        outputs = self.forward(sample).flatten()
        loss = self.criterion(outputs, labels)
        loss = loss.mean()

        test_dict = {}
        test_dict['labels'] = labels
        test_dict['outputs'] = outputs
        test_dict['masses'] = sample['mass'].float()
        test_dict['index'] = sample['index'].float()

        if not self.supervised:
            test_dict['charges'] = sample['charge'].float()
            test_dict['true_labels'] = labels
        else:
            test_dict['charges'] = labels
            test_dict['true_labels'] = sample['true_label'].float()
        
        return test_dict
        
    def test_epoch_end(self, dict_list):
        full_dict = {k: [] for k in dict_list[0].keys()}
        for d in dict_list:
            for k in full_dict.keys():
                full_dict[k].append(d[k])
        for k in full_dict.keys():
            full_dict[k] = torch.cat(full_dict[k]).cpu().detach().numpy().squeeze()

        # Save test values to files
        self.test_labels = full_dict['labels']
        self.test_outs = full_dict['outputs']
        self.test_idx = full_dict['index']
        self.test_masses = full_dict['masses']
        if self.supervised:
            self.test_true_labels = full_dict['true_labels']
            np.save(f'{self.version_dir}/test_truelabels_{self.device}.npy', self.test_true_labels)
        np.save(f'{self.version_dir}/test_labels_{self.device}.npy', self.test_labels)
        np.save(f'{self.version_dir}/test_outs_{self.device}.npy', self.test_outs)
        np.save(f'{self.version_dir}/test_idx_{self.device}.npy', self.test_idx)
        np.save(f'{self.version_dir}/test_masses_{self.device}.npy', self.test_masses)
        
        # Evaluate performance on full test set
        if not self.supervised:
            self.fpr, self.tpr, thresh = calc_weak_roc_splot(full_dict['masses'],
                full_dict['outputs'])
        else:
            self.fpr, self.tpr, thresh = roc_curve(full_dict['labels'], full_dict['outputs'])
        self.opt_thresh = thresh[np.argmax(self.tpr - self.fpr)]
        self.auc = auc(self.fpr,self.tpr)
        self.opt_thresh_acc = np.sum(np.int32(
            full_dict['outputs']>=self.opt_thresh)==full_dict['labels'])/len(full_dict['outputs'])
        self.log('auc', self.auc, sync_dist=True)
        self.log('test_acc', self.opt_thresh_acc, sync_dist=True)
        
class iso_net(base_classifier_net):
    def __init__(self, num_cones, layer_size, learning_rate, version_dir,
        supervised, patience, min_change, batch_size):
        super().__init__(learning_rate, version_dir, supervised=supervised,
            patience=patience, min_change=min_change, batch_size=batch_size)

        self.num_cones = num_cones
        self.layer_size = layer_size
        
        self.save_hyperparameters('layer_size')
        
        # Configure network
        self.network = nn.Sequential(
            nn.Linear(self.num_cones, self.layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.layer_size,self.layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.layer_size,self.layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.layer_size,1),
            nn.Sigmoid())
        
    def forward(self, x):
        out = self.network(x['input'].float())
        return out
        
class pfn(base_classifier_net):
    def __init__(self, Phi_size, F_size, l_size, learning_rate, 
        version_dir, patience=None, min_change=None, pids=True, supervised=False,
        batch_size=None):
        super().__init__(learning_rate, version_dir, supervised=supervised,
            patience=patience, min_change=min_change, batch_size=batch_size)

        self.Phi_size = Phi_size
        self.F_size = F_size
        self.l_size = l_size
        if pids:
            self.num_feats = 4
        else:
            self.num_feats = 3
        
        self.save_hyperparameters('Phi_size')
        self.save_hyperparameters('F_size')
        self.save_hyperparameters('l_size')
        
        # Configure network
        self.input_layer = nn.Linear(self.num_feats, self.Phi_size)

        self.Phi_net = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.Phi_size, self.Phi_size),
            nn.LeakyReLU(),
            nn.Linear(self.Phi_size, self.Phi_size),
            nn.LeakyReLU(),
            nn.Linear(self.Phi_size, self.l_size))

        self.F_net = nn.Sequential(
            nn.Linear(self.l_size, self.F_size),
            nn.LeakyReLU(),
            nn.Linear(self.F_size, self.F_size),
            nn.LeakyReLU(),
            nn.Linear(self.F_size, self.F_size),
            nn.LeakyReLU(),
            nn.Linear(self.F_size, self.F_size),
            nn.LeakyReLU(),
            nn.Linear(self.F_size, 1),
            nn.Sigmoid())

    def forward(self, x):
        out = self.input_layer(x['input'].float())
        out = self.Phi_net(out)
        out = torch.sum(out, dim=1)
        out = self.F_net(out)
        return out

class iso_efp_net(base_classifier_net):
    def __init__(self, num_cones, num_efps, layer_size, learning_rate, 
        version_dir, include_sum_pT=True, supervised=False, patience=None,
        min_change=None, batch_size=None):
        super().__init__(learning_rate, version_dir, supervised=supervised,
            patience=patience, min_change=min_change, batch_size=batch_size)

        self.num_cones = num_cones
        self.num_efps = num_efps
        if include_sum_pT:
            self.input_size = self.num_cones + self.num_efps + 1
        else:
            self.input_size = self.num_cones + self.num_efps
        self.layer_size = layer_size
        
        self.save_hyperparameters('layer_size')
        
        # Configure network
        self.network = nn.Sequential(
            nn.Linear(self.input_size, self.layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.layer_size,self.layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.layer_size,self.layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.layer_size,1),
            nn.Sigmoid())
        
    def forward(self, x):
        out = self.network(x['input'].float().squeeze())
        return out