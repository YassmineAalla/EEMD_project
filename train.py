# train.py
import numpy as np
import pandas as pd
import ast
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
from pre_process import extract_features
from config import DATA_PATH, MODEL_PATH, RF_TREES

def load_features(csv_path):
    print("Extracting features (this takes a few minutes)...")
    df = pd.read_csv(csv_path)
    X, y = [], []
    for i, row in df.iterrows():
        sig = np.array(ast.literal_eval(row['signal']))
        feats = extract_features(sig)
        X.append(feats)
        y.append(int(row['label']))
        if i % 50 == 0:
            print(f"  {i}/{len(df)} windows processed")
    return np.array(X), np.array(y)

def select_features(X_train, y_train):
    print("Running LASSO feature selection...")
    lasso = LassoCV(cv=5, max_iter=10000, random_state=42)
    lasso.fit(X_train, y_train)
    selected = np.where(lasso.coef_ != 0)[0]
    if len(selected) < 5:           # fallback: keep top 10 by coef magnitude
        selected = np.argsort(np.abs(lasso.coef_))[-10:]
    print(f"  Selected {len(selected)} features from {X_train.shape[1]}")
    return selected

def train(csv_path=DATA_PATH):
    X, y = load_features(csv_path)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    selected = select_features(X_tr, y_tr)
    X_tr_sel = X_tr[:, selected]
    X_te_sel  = X_te[:,  selected]

    pipe = Pipeline([
        ('sampler', ADASYN(random_state=42)),
        ('clf', RandomForestClassifier(
            n_estimators=RF_TREES,
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
    joblib.dump({'model': pipe, 'features': selected}, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")

if __name__ == '__main__':
    train()