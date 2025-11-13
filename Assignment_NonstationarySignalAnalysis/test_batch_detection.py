# test_batch_detection.py
import glob, json, os
import numpy as np
from utils_io import try_load_record
from Pan_Tompkins import detect_r_peaks_with_fallback
from scipy.signal import welch

def bandpow(x, fs, fmin, fmax):
    f,P = welch(x, fs=fs, nperseg=1024)
    return float(P[(f>=fmin)&(f<=fmax)].sum())

out = []
for p in sorted(glob.glob("a000*.dat")):
    base = os.path.splitext(p)[0]
    try:
        info = try_load_record(base)
        sig = info['p_signal']; fs = info['fs']; chmap = info['channel_map']
        ecg_idx = chmap['ecg']; pcg_idx = chmap['pcg']
        res = {'file': os.path.basename(p), 'shape': sig.shape, 'fs': fs, 'ecg_idx':ecg_idx, 'pcg_idx':pcg_idx}
        ch_ecg = sig[:, ecg_idx]
        ch_pcg = sig[:, pcg_idx]
        res['bp_ecg_ecgband'] = bandpow(ch_ecg, fs, 5, 40)
        res['bp_pcg_ecgband'] = bandpow(ch_pcg, fs, 5, 40)
        # detect
        r = detect_r_peaks_with_fallback(ch_ecg, fs=fs, debug=False)
        res['n_r_peaks'] = int(len(r))
        # median RR (s)
        if len(r) >= 2:
            rrs = np.diff(r)/fs
            res['median_rr_s'] = float(np.median(rrs))
        else:
            res['median_rr_s'] = None
        out.append(res)
    except Exception as e:
        out.append({'file': os.path.basename(p), 'error': str(e)})
# write summary
with open("batch_detection_summary.json", "w") as f:
    json.dump(out, f, indent=2)
print("Wrote batch_detection_summary.json")
