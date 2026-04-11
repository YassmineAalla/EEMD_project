# train.py
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
from pre_process import extract_features
from config import DATASET_PATH, SENSOR, N_SAMPLES, RF_TREES, MODEL_PATH  # ← ajout RF_TREES et MODEL_PATH


def load_features_from_dataset():
    X, y = [], []
    skipped = 0

    # chemin vers le sous-dossier du capteur choisi
    sensor_folder = {
        'A1': os.path.join(DATASET_PATH, 'Accelerometer', 'Accelerometer'),
        'A2': os.path.join(DATASET_PATH, 'Accelerometer', 'Accelerometer'),
        'H1': os.path.join(DATASET_PATH, 'Hydrophone',    'Hydrophone'),
        'H2': os.path.join(DATASET_PATH, 'Hydrophone',    'Hydrophone'),
        'P1': os.path.join(DATASET_PATH, 'Dynamic Pressure Sensor', 'Dynamic Pressure Sensor'),
        'P2': os.path.join(DATASET_PATH, 'Dynamic Pressure Sensor', 'Dynamic Pressure Sensor'),
    }
    search_path = sensor_folder.get(SENSOR, DATASET_PATH)

    # parcourt récursivement tous les sous-dossiers
    for root, dirs, files in os.walk(search_path):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)

            # CSV (accéléromètre ou pression)
            if fname.endswith('.csv') and SENSOR in fname:
                try:
                    df     = pd.read_csv(fpath)
                    signal = df['Value'].dropna().values.astype(float)

                    for start in range(0, len(signal) - N_SAMPLES, N_SAMPLES):
                        window = signal[start:start + N_SAMPLES]
                        feats  = extract_features(window)
                        X.append(feats)
                        y.append(_label_from_filename(fname))

                    print(f"  OK {fname} → {len(signal)//N_SAMPLES} fenêtres")

                except Exception as e:
                    print(f"  Skipped {fname}: {e}")
                    skipped += 1

            # RAW (hydrophone)
            elif fname.endswith('.raw') and SENSOR in fname:
                try:
                    raw = np.fromfile(fpath, dtype=np.int16).astype(float)
                    for start in range(0, len(raw) - N_SAMPLES, N_SAMPLES):
                        window = raw[start:start + N_SAMPLES]
                        feats  = extract_features(window)
                        X.append(feats)
                        y.append(_label_from_filename(fname))

                    print(f"  OK {fname} → {len(raw)//N_SAMPLES} fenêtres")

                except Exception as e:
                    print(f"  Skipped {fname}: {e}")
                    skipped += 1

    y = np.array(y)
    X = np.array(X)

    # filtre labels inconnus
    mask = y != -1
    X, y = X[mask], y[mask]

    print(f"\nDataset chargé : {len(X)} fenêtres")
    print(f"  Leak    : {sum(y)}")
    print(f"  No-leak : {len(y) - sum(y)}")
    print(f"  Skipped : {skipped} fichiers")
    return X, y
def _label_from_filename(fname):
    # extrait le code entre le 2ème et 3ème underscore
    # BR_LC_0.18 LPS_A1.csv → partie[1] = 'LC'
    # LO_NL_ND_A1.csv       → partie[1] = 'NL'
    try:
        parts = fname.split('_')
        code  = parts[1].upper()   # LC, CC, GL, OL, ou NL
    except IndexError:
        print(f"  WARNING nom inattendu : {fname}")
        return -1

    if code in ['LC', 'CC', 'GL', 'OL']:
        return 1   # leak
    if code == 'NL':
        return 0   # no leak

    print(f"  WARNING label inconnu : {fname} (code={code})")
    return -1
def select_features(X_train, y_train):
    print("Running LASSO feature selection...")
    lasso = LassoCV(cv=5, max_iter=10000, random_state=42)
    lasso.fit(X_train, y_train)
    selected = np.where(lasso.coef_ != 0)[0]
    if len(selected) < 5:
        selected = np.argsort(np.abs(lasso.coef_))[-10:]
    print(f"  Selected {len(selected)} features from {X_train.shape[1]}")
    return selected


def train():
    X, y = load_features_from_dataset()   # ← correction : une seule paire de parenthèses

    # filtre les labels inconnus (-1)
    mask = y != -1
    X, y = X[mask], y[mask]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    selected = select_features(X_tr, y_tr)
    X_tr_sel = X_tr[:, selected]
    X_te_sel = X_te[:, selected]

    pipe = Pipeline([
        ('sampler', ADASYN(random_state=42)),
        ('clf', RandomForestClassifier(
            n_estimators=RF_TREES,       # ← vient maintenant de config.py
            class_weight='balanced',
            n_jobs=-1,
            random_state=42))
    ])

    print("Training Random Forest...")
    pipe.fit(X_tr_sel, y_tr)

    y_pred = pipe.predict(X_te_sel)
    print("\n--- Evaluation ---")
    print(classification_report(y_te, y_pred, target_names=['No leak', 'Leak']))

    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': pipe, 'features': selected}, MODEL_PATH)  # ← vient de config.py
    print(f"Model saved → {MODEL_PATH}")


if __name__ == '__main__':
    train()