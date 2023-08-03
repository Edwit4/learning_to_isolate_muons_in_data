import numpy as np
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

def exp_dist(x, tau):
    return np.exp(-x/tau)

def voigt_exp_dist(x, tau, alpha, gamma, s):
    voigt = voigt_profile(x-90.8, alpha, gamma)
    voigt = voigt / np.trapz(voigt,x=x)
    exp = exp_dist(x, tau)
    exp = exp / np.trapz(exp,x=x)
    return s*voigt + (1.-s)*exp

def fit_Ms(fit_masses, initial_guess):
    # Determine appropriate number of bins
    q75, q25 = np.percentile(fit_masses, [75 ,25])
    iqr = q75 - q25
    bin_width = 2*iqr*len(fit_masses)**(-1/3)
    num_bins = int((np.max(fit_masses) - np.min(fit_masses)) / bin_width)

    # Make histogram
    mass_hist, bin_edges = np.histogram(fit_masses, bins=num_bins, density=True)

    # Get values at centers of bins
    window_centered = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Perform fit
    # popt in order: (0) exp_lam, (1) exp_const, (2) exp_scale, (3) voigt_sig, (4) voigt_gam, (5) voigt_mass, (6) norm, (7) sig_frac
    bounds_lo = (0, 0, 0, 0)
    bounds_hi = (50.0, 2, 2, 1)
    popt, pcov = curve_fit(voigt_exp_dist, window_centered, mass_hist, p0=initial_guess, 
                           bounds=(bounds_lo, bounds_hi), maxfev=5000)

    # Calculate error on fit parameters
    perr = np.sqrt(np.diag(pcov))
    
    return popt, perr

