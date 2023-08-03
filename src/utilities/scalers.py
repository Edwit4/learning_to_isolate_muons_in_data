import torch
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from src.utilities.data_utils import atleast_2d

class pfn_standard_scaler(TransformerMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pT_scaler = StandardScaler()
        self.eta_scaler = StandardScaler()
        self.phi_scaler = StandardScaler()

    def fit(self, data):
        pT = data[...,0].flatten()
        nonzeros = (pT!=0)
        pT = atleast_2d(pT[nonzeros]).T
        eta = atleast_2d(data[...,1].flatten()[nonzeros]).T
        phi = atleast_2d(data[...,2].flatten()[nonzeros]).T
        self.pT_scaler.fit(pT)
        self.eta_scaler.fit(eta)
        self.phi_scaler.fit(phi)

    def partial_fit(self, batch):
        pT = batch[...,0].flatten()
        nonzeros = (pT!=0)
        pT = atleast_2d(pT[nonzeros]).T
        eta = atleast_2d(batch[...,1].flatten()[nonzeros]).T
        phi = atleast_2d(batch[...,2].flatten()[nonzeros]).T
        self.pT_scaler.partial_fit(pT)
        self.eta_scaler.partial_fit(eta)
        self.phi_scaler.partial_fit(phi)

    def transform(self, data):
        pT_shape = data[...,0].shape
        eta_shape = data[...,0].shape
        phi_shape = data[...,0].shape
        if torch.is_tensor(data):
            data[...,0] = torch.from_numpy(
                self.pT_scaler.transform(atleast_2d(data[...,0].flatten()).T).reshape(pT_shape))
            data[...,1] = torch.from_numpy(
                self.eta_scaler.transform(atleast_2d(data[...,1].flatten()).T).reshape(eta_shape))
            data[...,2] = torch.from_numpy(
                self.phi_scaler.transform(atleast_2d(data[...,2].flatten()).T).reshape(phi_shape))
        else:
            data[...,0] = self.pT_scaler.transform(atleast_2d(data[...,0].flatten()).T).reshape(pT_shape)
            data[...,1] = self.eta_scaler.transform(atleast_2d(data[...,1].flatten()).T).reshape(eta_shape)
            data[...,2] = self.phi_scaler.transform(atleast_2d(data[...,2].flatten()).T).reshape(phi_shape)
        return data

    def get_params(self):
        return self.pT_scaler.get_params(), self.eta_scaler.get_params(), self.phi_scaler.get_params()