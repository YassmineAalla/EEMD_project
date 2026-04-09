# predict.py
import numpy as np
import joblib
import time
from pre_process import extract_features
from simulate_data import make_background, make_leak_signal
from config import MODEL_PATH, N_SAMPLES, LEAK_THRESHOLD

def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle['model'], bundle['features']

def read_window_simulated(inject_leak=False):
    """Replace this function with real ADC reading on deployment."""
    sig = make_background(N_SAMPLES)
    if inject_leak:
        sig += make_leak_signal(N_SAMPLES, severity=1.2)
    return sig

def predict_window(signal, model, selected_features):
    feats = extract_features(signal)
    feats_sel = feats[selected_features].reshape(1, -1)
    proba = model.predict_proba(feats_sel)[0][1]
    return proba >= LEAK_THRESHOLD, round(float(proba), 3)

def run(n_windows=20):
    model, selected = load_model()
    print(f"Model loaded. Running inference on {n_windows} windows...\n")
    for i in range(n_windows):
        inject = (i % 7 == 0)           # inject a leak every 7 windows
        signal = read_window_simulated(inject_leak=inject)
        t0 = time.time()
        is_leak, prob = predict_window(signal, model, selected)
        elapsed = round((time.time() - t0) * 1000, 1)
        status = "LEAK DETECTED" if is_leak else "clear"
        marker = " <--" if inject else ""
        print(f"Window {i+1:02d} | {status:15s} | p={prob:.3f} | {elapsed}ms{marker}")
        time.sleep(0.1)

if __name__ == '__main__':
    run()