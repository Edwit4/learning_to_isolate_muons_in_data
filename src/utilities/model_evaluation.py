import numpy as np
import pandas as pd
from scipy.special import voigt_profile
from hep_ml import splot
from src.utilities.fitting import fit_Ms, exp_dist

def calc_weak_roc_splot(masses, outs, num_bins=None):

    sort = np.argsort(masses)
    masses = masses[sort]
    outs = outs[sort]
    
    initial_guess = [15, 1.5, 1.1, 0.8]
    popt, perr = fit_Ms(masses, initial_guess)
    
    if num_bins is None:
        q75, q25 = np.percentile(outs, [75 ,25])
        iqr = q75 - q25
        bin_width = (2*iqr*((1-popt[-1])*len(outs))**(-1/3))
        num_bins = int(np.ceil((np.max(outs) - np.min(outs)) / bin_width))
    
    exp = exp_dist(masses, popt[0])
    voigt = voigt_profile(masses - 90.8, popt[1], popt[2])
    exp = (1.-popt[-1])*(exp / np.trapz(exp, x=masses))
    voigt = popt[-1]*(voigt / np.trapz(voigt, x=masses))
    
    voigt_probs = voigt / (voigt + exp)
    exp_probs = exp / (voigt + exp)
    
    probs = pd.DataFrame(dict(sig=voigt_probs, bck=exp_probs))
    if np.allclose(probs.sum(axis=1), 1, atol=1e-3):
        sWeights = splot.compute_sweights(probs)
    
        sig_hist, bin_edges = np.histogram(outs, weights=sWeights.sig, bins=num_bins)
        bg_hist, _ = np.histogram(outs, weights=sWeights.bck, bins=bin_edges)
    
        sig_hist[sig_hist<0] = 0
        bg_hist[bg_hist<0] = 0
        
        fpr = []
        tpr = []
        for i,t in enumerate(bin_edges):
            fpr.append(np.sum(bg_hist[i:])/np.sum(bg_hist))
            tpr.append(np.sum(sig_hist[i:])/np.sum(sig_hist))
            
        _, unique_i = np.unique(fpr,return_index=True)
        unique_i = unique_i[::-1]
        fpr = np.array(fpr)[unique_i]
        tpr = np.array(tpr)[unique_i]
        bin_edges = np.array(bin_edges)[unique_i]
        
    else:
        fpr = np.array([0,0.5,1])
        tpr = np.array([0,0.5,1])
        bin_edges = np.array([0,0.5,1])
        
    return fpr, tpr, bin_edges
