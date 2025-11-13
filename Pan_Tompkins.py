# Pan_Tompkins.py (patched: amplitude stabilization added)
"""
Pan_Tompkins.py

NumPy-only version of the Pan-Tompkins style QRS detector.
Replaces SciPy dependencies (butter, filtfilt, find_peaks) with
pure-NumPy equivalents where needed.

Changes in this patched version:
 - Adds amplitude stabilization (_amplitude_stabilize) to suppress
   extreme transients / abnormal amplitude sections that corrupt
   adaptive thresholds. This is applied before filtering in both
   detect_r_peaks(...) and pt_pipeline(...).
 - API, function names, parameters, and return values preserved.
"""
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

# ---------------- filtering / preprocessing helpers ----------------
def _design_bandpass_fir(fs: int, low: float, high: float, kernel_len: Optional[int] = None) -> np.ndarray:
    if kernel_len is None:
        kernel_len = int(round(min(1025, max(31, 0.128 * fs))))
    kernel_len = int(kernel_len)
    if kernel_len % 2 == 0:
        kernel_len += 1
    nyq = 0.5 * fs
    f1 = float(low) / nyq
    f2 = float(high) / nyq
    m = (kernel_len - 1) // 2
    n = np.arange(-m, m+1, dtype=float)
    h = (np.sinc(2 * f2 * n) - np.sinc(2 * f1 * n))
    w = 0.5 * (1.0 - np.cos(2.0 * np.pi * (np.arange(kernel_len) / float(kernel_len - 1))))
    h *= w
    denom = np.sum(np.abs(h))
    if denom <= 0:
        denom = 1.0
    h /= denom
    return h

def _filtfilt_like_linear_phase(sig: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    sig = np.asarray(sig, dtype=float)
    if sig.size == 0 or kernel.size == 0:
        return sig.copy()
    pad = len(kernel)
    if sig.size >= pad:
        left = sig[:pad][::-1]
        right = sig[-pad:][::-1]
    else:
        left = sig[::-1]
        right = sig[::-1]
    xpad = np.concatenate([left, sig, right])
    y = np.convolve(xpad, kernel, mode='same')
    start = pad
    end = start + len(sig)
    return y[start:end]

def _five_point_derivative(sig: np.ndarray, fs: int) -> np.ndarray:
    kernel = np.array([-1.0, -2.0, 0.0, 2.0, 1.0])
    deriv = np.convolve(sig, kernel, mode='same')
    deriv *= (fs / 8.0)
    return deriv

def _moving_window_integration(x: np.ndarray, fs: int, window_ms: float = 150.0) -> np.ndarray:
    win = max(1, int(round(window_ms / 1000.0 * fs)))
    kernel = np.ones(win) / float(win)
    return np.convolve(x, kernel, mode='same')

def _refine_to_ecg_peak(ecg: np.ndarray, center_idx: int, fs: int, rad_ms: float = 30.0) -> int:
    rad = max(1, int(round(rad_ms / 1000.0 * fs)))
    lo = max(0, center_idx - rad)
    hi = min(len(ecg) - 1, center_idx + rad)
    segment = ecg[lo:hi+1]
    if segment.size == 0:
        return center_idx
    arg = int(np.argmax(np.abs(segment)))
    return lo + arg

def _max_slope_around(ecg: np.ndarray, idx: int, fs: int, window_ms: int = 40) -> float:
    rad = max(1, int(round(window_ms / 1000.0 * fs)))
    lo = max(0, idx - rad)
    hi = min(len(ecg) - 1, idx + rad)
    seg = ecg[lo:hi+1]
    if seg.size < 2:
        return 0.0
    slopes = np.abs(np.diff(seg))
    return float(np.max(slopes))

def _find_peaks_numpy(x: np.ndarray, height: Optional[float] = None, distance: Optional[int] = None) -> np.ndarray:
    x = np.asarray(x)
    N = x.size
    if N == 0:
        return np.array([], dtype=int)
    left = np.r_[x[0] - 1e-12, x[:-1]]
    right = np.r_[x[1:], x[-1] - 1e-12]
    peaks_mask = (x > left) & (x >= right)
    peaks_idx = np.nonzero(peaks_mask)[0]
    if height is not None:
        peaks_idx = peaks_idx[x[peaks_idx] >= height]
    if distance is None or distance <= 1 or peaks_idx.size <= 1:
        return peaks_idx.astype(int)
    peaks_vals = x[peaks_idx]
    order = np.argsort(-peaks_vals)
    keep = np.zeros_like(peaks_idx, dtype=bool)
    taken = np.zeros(N, dtype=bool)
    for i in order:
        idx = peaks_idx[i]
        if not taken[idx]:
            keep[i] = True
            lo = max(0, idx - distance)
            hi = min(N, idx + distance + 1)
            taken[lo:hi] = True
    kept = peaks_idx[keep]
    kept.sort()
    return kept.astype(int)

# ---------------- robust normalize / fallback ----------------
def _robust_normalize(x):
    x = np.asarray(x).astype(float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-6:
        std = np.std(x) if np.std(x) > 1e-6 else 1.0
        return (x - med) / std
    return (x - med) / (1.4826 * mad)

def _fallback_qrs_detector(ecg, fs):
    low, high = 1.0, 40.0
    try:
        kernel = _design_bandpass_fir(fs, low, high, kernel_len=None)
        filtered = _filtfilt_like_linear_phase(ecg, kernel)
    except Exception:
        filtered = ecg
    env = np.abs(filtered)
    med = np.median(env)
    thresh = med + 0.5 * np.std(env)
    distance = int(round(0.18 * fs))
    peaks = _find_peaks_numpy(env, height=thresh, distance=distance)
    return peaks

# ---------------- amplitude stabilization ----------------
def _amplitude_stabilize(ecg: np.ndarray, fs: int, env_win_ms: float = 200.0,
                         max_ratio_thresh: float = 8.0,
                         gain_clip: tuple = (0.3, 3.0)) -> (np.ndarray, bool):
    """
    Adaptive amplitude stabilization:
     - compute envelope E = moving average(|ecg|, env_win_ms)
     - if max(E)/median(E) > max_ratio_thresh -> compute gain = median(E) / (E + eps)
       and clip gain into gain_clip range to avoid extreme distortion.
     - return (ecg * gain, changed_flag)
    If no severe transient, returns original ecg and False.
    """
    ecg = np.asarray(ecg, dtype=float)
    win = max(1, int(round(env_win_ms / 1000.0 * fs)))
    kernel = np.ones(win) / float(win)
    E = np.convolve(np.abs(ecg), kernel, mode='same')
    med = np.median(E) if E.size > 0 else 0.0
    eps = 1e-12
    if med <= 0:
        med = np.mean(E) + 1e-9
    ratio = np.max(E) / (med + eps) if E.size > 0 else 1.0
    if ratio <= max_ratio_thresh:
        return ecg, False
    # compute gain and clip
    gain = med / (E + eps)
    lo_clip, hi_clip = gain_clip
    gain = np.clip(gain, lo_clip, hi_clip)
    ecg_stab = ecg * gain
    return ecg_stab, True

# ---------------- main detector ----------------
def detect_r_peaks_with_fallback(ecg, fs=2000, debug=False):
    ecg_norm = _robust_normalize(ecg)
    try:
        r = detect_r_peaks(ecg_norm, fs=fs, debug=debug)
    except Exception:
        r = np.array([], dtype=int)
    if r is None or len(r) < 2:
        fb = _fallback_qrs_detector(ecg_norm, fs)
        if len(fb) >= 2:
            refined = []
            for p in fb:
                lo = max(0, p - int(0.02 * fs))
                hi = min(len(ecg_norm) - 1, p + int(0.02 * fs))
                sub = ecg_norm[lo:hi+1]
                if sub.size == 0:
                    continue
                arg = int(np.argmax(np.abs(sub)))
                refined.append(lo + arg)
            r = np.unique(np.array(refined, dtype=int))
    if r is None:
        return np.array([], dtype=int)
    return np.array(sorted(np.unique(r)), dtype=int)

def detect_r_peaks(ecg: np.ndarray,
                   fs: int = 2000,
                   low_hz: float = 5.0,
                   high_hz: float = 15.0,
                   integration_ms: float = 150.0,
                   refractory_ms: float = 200.0,
                   search_back_factor: float = 1.66,
                   debug: bool = False) -> np.ndarray:
    """
    Pan-Tompkins style QRS detector implemented using NumPy-only primitives.
    This version applies amplitude stabilization first to prevent early transients
    from dominating the adaptive thresholds.
    """
    ecg = np.asarray(ecg).ravel()
    N = ecg.size
    if N == 0:
        return np.array([], dtype=int)

    # --- amplitude stabilization: suppress large early transients if present ---
    try:
        ecg_proc, changed = _amplitude_stabilize(ecg, fs, env_win_ms=200.0, max_ratio_thresh=8.0, gain_clip=(0.3, 3.0))
        if changed and debug:
            print("[PT] amplitude stabilization applied (transient suppressed)")
    except Exception:
        ecg_proc = ecg.copy()

    # 1) bandpass filter (replace butter+filtfilt)
    try:
        kernel = _design_bandpass_fir(fs, low_hz, high_hz, kernel_len=None)
        ecg_f = _filtfilt_like_linear_phase(ecg_proc, kernel)
    except Exception:
        ecg_f = ecg_proc.copy()

    # 2) derivative
    deriv = _five_point_derivative(ecg_f, fs)

    # 3) squaring
    squared = deriv ** 2

    # 4) moving window integration
    mwi = _moving_window_integration(squared, fs, integration_ms)

    # gather candidate peaks from integrated signal
    min_dist = max(1, int(round(refractory_ms / 1000.0 * fs)))
    cand_idxs = _find_peaks_numpy(mwi, distance=min_dist)

    if cand_idxs.size == 0:
        if debug:
            print("[PT] No MWI peaks found; returning empty array")
        return np.array([], dtype=int)

    cand_vals_i = mwi[cand_idxs]
    cand_vals_f = np.abs(ecg_f[cand_idxs])

    # Initialization using first segment
    n_init = min(8, cand_idxs.size)
    init_idxs = cand_idxs[:n_init]
    init_i = mwi[init_idxs] if init_idxs.size > 0 else np.array([0.0])
    init_f = np.abs(ecg_f[init_idxs]) if init_idxs.size > 0 else np.array([0.0])

    SPKI = float(np.max(init_i)) if init_i.size > 0 else 1.0
    NPKI = float(np.median(init_i) * 0.5) if init_i.size > 0 else SPKI * 0.125
    SPKF = float(np.max(init_f)) if init_f.size > 0 else 1.0
    NPKF = float(np.median(init_f) * 0.5) if init_f.size > 0 else SPKF * 0.125

    THRESHOLD_I = NPKI + 0.25 * (SPKI - NPKI)
    THRESHOLD_F = NPKF + 0.25 * (SPKF - NPKF)
    THRESHOLD_I2 = 0.5 * THRESHOLD_I
    THRESHOLD_F2 = 0.5 * THRESHOLD_F

    detected = []
    detected_i_vals = []
    detected_f_vals = []
    rr_intervals = []

    for idx_cand in cand_idxs:
        val_i = float(mwi[idx_cand])
        val_f = float(np.abs(ecg_f[idx_cand]))

        # enforce refractory
        if detected and (idx_cand - detected[-1]) < min_dist:
            NPKI = 0.125 * val_i + 0.875 * NPKI
            NPKF = 0.125 * val_f + 0.875 * NPKF
            THRESHOLD_I = NPKI + 0.25 * (SPKI - NPKI)
            THRESHOLD_F = NPKF + 0.25 * (SPKF - NPKF)
            THRESHOLD_I2 = 0.5 * THRESHOLD_I
            THRESHOLD_F2 = 0.5 * THRESHOLD_F
            continue

        accept = False
        threshold_used = None

        if (val_i >= THRESHOLD_I) or (val_f >= THRESHOLD_F):
            accept = True
            threshold_used = 'primary'
        elif (val_i >= THRESHOLD_I2) or (val_f >= THRESHOLD_F2):
            accept = True
            threshold_used = 'secondary'

        if accept:
            refined = _refine_to_ecg_peak(ecg, idx_cand, fs, rad_ms=30.0)

            is_t_wave = False
            if detected:
                prev = detected[-1]
                dt = (refined - prev) / float(fs)
                if dt < 0.36:
                    curr_slope = _max_slope_around(ecg, refined, fs)
                    prev_slope = _max_slope_around(ecg, prev, fs)
                    if prev_slope > 0 and curr_slope < 0.5 * prev_slope:
                        is_t_wave = True

            if is_t_wave:
                NPKI = 0.125 * val_i + 0.875 * NPKI
                NPKF = 0.125 * val_f + 0.875 * NPKF
            else:
                if not detected or (refined - detected[-1]) >= min_dist:
                    detected.append(int(refined))
                    detected_i_vals.append(val_i)
                    detected_f_vals.append(val_f)
                    if threshold_used == 'primary':
                        SPKI = 0.125 * val_i + 0.875 * SPKI
                        SPKF = 0.125 * val_f + 0.875 * SPKF
                    else:
                        SPKI = 0.25 * val_i + 0.75 * SPKI
                        SPKF = 0.25 * val_f + 0.75 * SPKF
                    if len(detected) >= 2:
                        rr = detected[-1] - detected[-2]
                        rr_intervals.append(rr)
                else:
                    NPKI = 0.125 * val_i + 0.875 * NPKI
                    NPKF = 0.125 * val_f + 0.875 * NPKF
        else:
            NPKI = 0.125 * val_i + 0.875 * NPKI
            NPKF = 0.125 * val_f + 0.875 * NPKF

        THRESHOLD_I = NPKI + 0.25 * (SPKI - NPKI)
        THRESHOLD_F = NPKF + 0.25 * (SPKF - NPKF)
        THRESHOLD_I2 = 0.5 * THRESHOLD_I
        THRESHOLD_F2 = 0.5 * THRESHOLD_F

    detected = np.array(detected, dtype=int)

    # Search-back for missed beats
    if detected.size >= 2:
        rr_ms = np.diff(detected)
        if rr_ms.size > 0:
            RRavg = int(np.mean(rr_ms[-8:]))
            if RRavg > 0:
                RRmiss = int(round(search_back_factor * RRavg))
                new_found = []
                for i in range(len(detected) - 1):
                    a = detected[i]; b = detected[i+1]
                    gap = b - a
                    if gap > RRmiss:
                        lo = a + 1; hi = b - 1
                        if lo >= hi:
                            continue
                        local_idx_rel = int(np.argmax(mwi[lo:hi+1]))
                        local_idx = lo + local_idx_rel
                        if mwi[local_idx] >= THRESHOLD_I2:
                            refined = _refine_to_ecg_peak(ecg, local_idx, fs, rad_ms=30.0)
                            new_found.append(int(refined))
                            SPKI = 0.25 * float(mwi[local_idx]) + 0.75 * SPKI
                            SPKF = 0.25 * float(np.abs(ecg_f[local_idx])) + 0.75 * SPKF
                            THRESHOLD_I = NPKI + 0.25 * (SPKI - NPKI)
                            THRESHOLD_F = NPKF + 0.25 * (SPKF - NPKF)
                if new_found:
                    detected = np.unique(np.concatenate([detected, np.array(new_found, dtype=int)]))
                    detected.sort()

    # Final pass: remove too-close peaks (within refractory) keeping the strongest ECG amplitude
    if detected.size > 1:
        final = []
        i = 0
        while i < detected.size:
            block = [detected[i]]
            j = i + 1
            while j < detected.size and (detected[j] - detected[i]) < min_dist:
                block.append(detected[j]); j += 1
            if len(block) == 1:
                final.append(block[0])
            else:
                vals = [abs(ecg[idx]) for idx in block]
                keep = block[int(np.argmax(vals))]
                final.append(keep)
            i = j
        detected = np.array(final, dtype=int)

    detected.sort()
    if debug:
        print(f"[Pan-Tompkins] final detected beats: {len(detected)}")
    return detected

# ---------------- pipeline / plotting (kept compatibility) ----------------
def pt_pipeline(ecg: np.ndarray, fs: int = 2000, low_hz: float = 5.0, high_hz: float = 15.0, integration_ms: float = 150.0):
    """
    Return intermediate arrays used by Pan-Tompkins:
      {
        'ecg_raw', 'ecg_filtered', 'deriv', 'squared', 'mwi', 'cand_idxs'
      }
    Applies amplitude stabilization so plotted pipeline matches detector input.
    """
    ecg = np.asarray(ecg).ravel()
    try:
        ecg_proc, _ = _amplitude_stabilize(ecg, fs, env_win_ms=200.0, max_ratio_thresh=8.0, gain_clip=(0.3, 3.0))
    except Exception:
        ecg_proc = ecg.copy()
    try:
        kernel = _design_bandpass_fir(fs, low_hz, high_hz, kernel_len=None)
        ecg_f = _filtfilt_like_linear_phase(ecg_proc, kernel)
    except Exception:
        ecg_f = ecg_proc.copy()
    kernel_der = np.array([-1., -2., 0., 2., 1.])
    deriv = np.convolve(ecg_f, kernel_der, mode='same') * (fs / 8.0)
    squared = deriv ** 2
    win = max(1, int(round(integration_ms / 1000.0 * fs)))
    mwi = np.convolve(squared, np.ones(win) / float(win), mode='same')
    distance = int(round(0.18 * fs))
    cand_idxs = _find_peaks_numpy(mwi, distance=distance)
    return {
        'ecg_raw': ecg,
        'ecg_filtered': ecg_f,
        'deriv': deriv,
        'squared': squared,
        'mwi': mwi,
        'cand_idxs': cand_idxs
    }

def plot_pt_pipeline(ecg: np.ndarray, fs: int = 2000, r_peaks: np.ndarray = None, figsize=(12, 9), show=True):
    data = pt_pipeline(ecg, fs)
    t = np.arange(len(ecg)) / float(fs)
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    axes[0].plot(t, data['ecg_raw'])
    if r_peaks is not None and len(r_peaks) > 0:
        axes[0].scatter(r_peaks / float(fs), data['ecg_raw'][r_peaks], c='r', s=12, zorder=3, label='R-peaks')
        axes[0].legend(fontsize='small')
    axes[0].set_ylabel("ECG (raw)")
    axes[1].plot(t, data['ecg_filtered'])
    axes[1].set_ylabel("Filtered")
    axes[2].plot(t, data['deriv'], label='derivative')
    axes[2].plot(t, data['squared'], label='squared', alpha=0.7)
    axes[2].legend(fontsize='small')
    axes[2].set_ylabel("Derivative / Squared")
    axes[3].plot(t, data['mwi'], label='MWI (windowed energy)')
    axes[3].scatter(data['cand_idxs'] / float(fs), data['mwi'][data['cand_idxs']], c='orange', s=10, label='candidates')
    axes[3].set_ylabel("MWI")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend(fontsize='small')
    plt.tight_layout()
    if show:
        plt.show()
    return fig
