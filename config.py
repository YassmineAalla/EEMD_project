# config.py
import os

DATASET_PATH = r'C:\Users\Yassmine\Desktop\pyflux-0.4.15\S5 CI1A\water leak acoustic detection\dataset_leaks\Dataset of Leak Simulations in Experimental Testbed Water Distribution System'

SENSOR       = 'A1'          # A1 ou A2 pour accéléromètre
FS           = 30780         # fréquence réelle calculée depuis les données
WINDOW_SEC   = 0.5
N_SAMPLES    = int(FS * WINDOW_SEC)   # = 15390 points par fenêtre

RF_TREES     = 200
MODEL_PATH   = 'models/leak_detector.pkl'

# ces variables ne sont plus utilisées mais gardées pour ne rien casser
N_IMFS                = 6
EEMD_TRIALS           = 50
EEMD_NOISE            = 0.2
LEAK_LABELS           = ['LC', 'CC', 'GL', 'OL']
N_FEATURES_PER_IMF    = 7
N_FEATURES_TOTAL      = N_IMFS * N_FEATURES_PER_IMF
LEAK_THRESHOLD        = 0.70
DATA_PATH             = 'data/simulated.csv'