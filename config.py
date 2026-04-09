# config.py

# config.py  —  APRÈS
FS          = 20_000
WINDOW_SEC  = 0.1                     # réduit à 0.1s
N_SAMPLES   = int(FS * WINDOW_SEC)   # = 2 000 points  ← OK
# --- EEMD ---
N_IMFS      = 6
EEMD_TRIALS = 20          # reduced from 100 to avoid memory errors
EEMD_NOISE  = 0.2         # fraction of signal std

# --- features ---
N_FEATURES_PER_IMF = 7    # energy, kurtosis, rms, std, crest, zcr, centroid
N_FEATURES_TOTAL   = N_IMFS * N_FEATURES_PER_IMF   # 42

# --- model ---
RF_TREES    = 200
LEAK_THRESHOLD = 0.70     # probability threshold for leak flag

# --- paths ---
DATA_PATH   = 'data/simulated.csv'
MODEL_PATH  = 'models/leak_detector.pkl'