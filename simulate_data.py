# simulate_data.py
import numpy as np
import pandas as pd
from config import FS, N_SAMPLES, DATA_PATH
import os

def make_background(n):
    """Pump hum + flow turbulence + broadband noise."""
    t = np.arange(n) / FS
    pump  = 0.4 * np.sin(2 * np.pi * 50 * t)     # 50 Hz pump fundamental
    pump += 0.2 * np.sin(2 * np.pi * 100 * t)     # 2nd harmonic
    flow  = 0.15 * np.random.randn(n)              # turbulent flow noise
    return pump + flow

def make_leak_signal(n, severity=1.0):
    """
    Leak acoustic signature: broadband burst with
    characteristic spectral peak around 300-800 Hz
    for plastic pipes, plus impulsive components.
    """
    t = np.arange(n) / FS
    # continuous leak hiss (band-limited noise)
    leak_noise = severity * 0.6 * np.random.randn(n)
    # bandpass around 500 Hz to mimic plastic pipe leak
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [300/(FS/2), 800/(FS/2)], btype='band')
    leak_hiss = filtfilt(b, a, leak_noise)
    # occasional impulsive bursts (crack/joint movement)
    impulses = np.zeros(n)
    n_bursts = np.random.randint(3, 8)
    for _ in range(n_bursts):
        idx = np.random.randint(0, n)
        width = np.random.randint(20, 80)
        burst = severity * 0.8 * np.random.randn(width)
        end = min(idx + width, n)
        impulses[idx:end] += burst[:end-idx]
    return leak_hiss + impulses

def simulate_dataset(n_windows=1000, leak_fraction=0.15, seed=42):
    np.random.seed(seed)
    rows = []
    for i in range(n_windows):
        is_leak = np.random.rand() < leak_fraction
        severity = np.random.uniform(0.5, 2.0) if is_leak else 0.0
        bg = make_background(N_SAMPLES)
        sig = bg + make_leak_signal(N_SAMPLES, severity) if is_leak else bg
        rows.append({'signal': sig.tolist(), 'label': int(is_leak),
                     'severity': round(severity, 2)})
    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {n_windows} windows → {DATA_PATH}")
    print(f"  Leak: {sum(r['label'] for r in rows)}  No-leak: {n_windows - sum(r['label'] for r in rows)}")
    return df

if __name__ == '__main__':
    simulate_dataset()