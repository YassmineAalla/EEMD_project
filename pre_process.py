# preprocess.py
import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis
from PyEMD import EEMD
from config import FS, N_IMFS, EEMD_TRIALS, EEMD_NOISE

# preprocess.py  —  fonction decompose(), APRÈS
def decompose(signal):
    sig_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-9)
    eemd = EEMD(trials=EEMD_TRIALS, noise_width=EEMD_NOISE)
    eemd.EMD.spline_kind = 'akima'
    eemd.EMD.boundary    = 'wave'
    eemd.EMD.FIXE_H      = 10
    eemd.noise_seed(42)
    eemd.parallel = False             # ← ligne ajoutée
    imfs = eemd.eemd(sig_norm)
    
    # Ensure exactly N_IMFS by padding or truncating
    if len(imfs) < N_IMFS:
        # Pad with zeros if fewer IMFs than expected
        padding = np.zeros((N_IMFS - len(imfs), len(signal)))
        imfs = np.vstack((imfs, padding))
    elif len(imfs) > N_IMFS:
        # Truncate if more IMFs than expected
        imfs = imfs[:N_IMFS]
    
    return imfs
def spectral_centroid(signal, fs):
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    psd_sum = np.sum(psd)
    if psd_sum == 0:
        return 0.0
    return float(np.sum(freqs * psd) / psd_sum)

def zero_crossing_rate(signal):
    return float(np.sum(np.diff(np.sign(signal)) != 0) / len(signal))

def crest_factor(signal):
    rms = np.sqrt(np.mean(signal**2))
    return float(np.max(np.abs(signal)) / (rms + 1e-9))

def extract_features(signal):
    """
    Returns a 1D feature vector of length N_IMFS * 7.
    Called once per window — this is your real-time hot path.
    """
    imfs = decompose(signal)
    features = []
    for imf in imfs:
        features += [
            float(np.sum(imf**2)),               # energy
            float(kurtosis(imf)),                 # impulsiveness
            float(np.sqrt(np.mean(imf**2))),      # RMS
            float(np.std(imf)),                   # std deviation
            crest_factor(imf),                    # peak / RMS
            zero_crossing_rate(imf),              # frequency proxy
            spectral_centroid(imf, FS),           # dominant frequency
        ]
    return np.array(features)