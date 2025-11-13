# utils_io.py
import os
import numpy as np
from typing import Dict, Any

def _bandpower(x, fs, fmin, fmax):
    # simple FFT-based bandpower
    n = len(x)
    if n < 2:
        return 0.0
    X = np.fft.rfft(x * np.hanning(n))
    ps = (np.abs(X) ** 2) / n
    freqs = np.fft.rfftfreq(n, 1.0/fs)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return float(ps[mask].sum())

def _count_sharp_peaks(x, fs):
    # quick heuristic: count peaks in derivative envelope that look QRS-like
    from scipy.signal import find_peaks
    dx = np.abs(np.diff(x, prepend=x[0]))
    env = np.convolve(dx, np.ones(max(1,int(round(0.01*fs))))/max(1,int(round(0.01*fs))), mode='same')
    # count peaks per minute-like normalization
    peaks, _ = find_peaks(env, height=np.median(env) + 0.5*np.std(env), distance=int(0.18*fs))
    return len(peaks)

def auto_detect_channels(p_signal: np.ndarray, fs: int = 2000) -> Dict[str,int]:
    """
    Heuristic to detect ECG vs PCG channel indices:
      - ECG has stronger bandpower in 5-40 Hz and more sharp peaks in derivative envelope.
      - PCG tends to have more energy above 50 Hz (but this depends).
    Returns {'ecg': idx, 'pcg': idx}
    """
    nch = p_signal.shape[1]
    # if only one channel, both map to 0
    if nch == 1:
        return {'ecg': 0, 'pcg': 0}
    scores = []
    for ch in range(nch):
        x = p_signal[:, ch].astype(float)
        bp_ecg = _bandpower(x, fs, 5, 40)   # QRS band
        bp_pcg = _bandpower(x, fs, 50, 300) # PCG band
        sharp_count = _count_sharp_peaks(x, fs)
        # score: prefer channel with higher relative ECG-band energy and more peaks
        score = (bp_ecg + 1e-9) / (bp_pcg + 1e-9) + 0.01 * sharp_count
        scores.append((ch, score, bp_ecg, bp_pcg, sharp_count))
    # sort by score descending
    scores.sort(key=lambda s: s[1], reverse=True)
    ecg_idx = scores[0][0]
    pcg_idx = scores[1][0] if scores[1][0] != ecg_idx else (0 if 0 != ecg_idx else 1)
    return {'ecg': int(ecg_idx), 'pcg': int(pcg_idx)}

def try_load_record(fname_or_base: str) -> Dict[str, Any]:
    """
    Attempt to load WFDB record or raw .dat.
    Returns dict: {'p_signal': ndarray, 'fs': int, 'sig_name': list, 'channel_map': {'ecg':i,'pcg':j}}
    """
    try:
        import wfdb
    except Exception:
        wfdb = None

    base, ext = os.path.splitext(fname_or_base)
    # try wfdb
    if wfdb is not None:
        try:
            rec = wfdb.rdrecord(base)
            p_signal = rec.p_signal.astype(float)
            fs = int(rec.fs)
            sig_name = [str(s) for s in rec.sig_name]
            # try to map ECG/PCG using names first
            ecg_idx = None; pcg_idx = None
            for i, nm in enumerate(sig_name):
                nm_up = nm.upper()
                if 'ECG' in nm_up or 'II' == nm_up or 'MLII' == nm_up:
                    ecg_idx = i
                if 'PCG' in nm_up or 'PHON' in nm_up or 'MIC' in nm_up:
                    pcg_idx = i
            if ecg_idx is None or pcg_idx is None:
                # fallback to auto detection
                chmap = auto_detect_channels(p_signal, fs)
            else:
                chmap = {'ecg': ecg_idx, 'pcg': pcg_idx}
            return {'p_signal': p_signal, 'fs': fs, 'sig_name': sig_name, 'channel_map': chmap}
        except Exception:
            pass

    # fallback to raw .dat
    dat_path = base + '.dat'
    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"No WFDB record and dat not found for base '{base}'")
    # try little then big endian
    raw = None
    for dtype in ('<i2', '>i2'):
        try:
            raw_try = np.fromfile(dat_path, dtype=dtype)
            if raw_try.size > 0:
                raw = raw_try
                break
        except Exception:
            raw = None
    if raw is None:
        raise ValueError("Cannot read dat as 16-bit ints")

    # assume 2-channel interleaved if possible
    if raw.size % 2 == 0:
        data = raw.reshape(-1, 2).astype(float)
    else:
        # try 1-channel
        data = raw.reshape(-1, 1).astype(float)

    fs = 2000
    # attempt auto-detection
    chmap = auto_detect_channels(data, fs)
    sig_name = ['CH0','CH1'][:data.shape[1]]
    return {'p_signal': data, 'fs': fs, 'sig_name': sig_name, 'channel_map': chmap}
