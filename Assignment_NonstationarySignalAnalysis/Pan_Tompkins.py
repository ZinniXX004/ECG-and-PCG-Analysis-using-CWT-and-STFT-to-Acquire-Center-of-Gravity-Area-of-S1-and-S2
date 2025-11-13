"""
Pan_Tompkins.py

Full Pan-Tompkins style QRS detector adapted from:
 - Pan & Tompkins (1985) algorithmic flow
 - additional practical initialization and T-wave discrimination heuristics
 - implementation inspired by the provided Pan_Tompkins.ipynb

Public API:
    detect_r_peaks(ecg: np.ndarray, fs: int = 2000, debug: bool=False) -> np.ndarray

Returns:
    numpy array of R-peak sample indices (sorted, unique)
"""

from typing import Optional
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

# ---------------- filtering / preprocessing helpers ----------------
def _butter_bandpass(sig: np.ndarray, fs: int, low: float = 5.0, high: float = 15.0, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, sig)


def _five_point_derivative(sig: np.ndarray, fs: int) -> np.ndarray:
    # Five-point derivative used in Pan-Tompkins; normalized by (1/(8T))
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
    hi = min(len(ecg)-1, center_idx + rad)
    segment = ecg[lo:hi+1]
    # choose maximal absolute value (robust to inverted ECG)
    arg = int(np.argmax(np.abs(segment)))
    return lo + arg


def _max_slope_around(ecg: np.ndarray, idx: int, fs: int, window_ms: int = 40) -> float:
    rad = max(1, int(round(window_ms / 1000.0 * fs)))
    lo = max(0, idx - rad)
    hi = min(len(ecg)-1, idx + rad)
    seg = ecg[lo:hi+1]
    if seg.size < 2:
        return 0.0
    slopes = np.abs(np.diff(seg))
    return float(np.max(slopes))


# ---------------- main detector ----------------
def _robust_normalize(x):
    x = np.asarray(x).astype(float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-6:
        std = np.std(x) if np.std(x) > 1e-6 else 1.0
        return (x - med) / std
    return (x - med) / (1.4826 * mad)  # approx. convert MAD to std

def _fallback_qrs_detector(ecg, fs):
    # fallback: bandpass 1-40 Hz then detect peaks in absolute filtered signal
    low, high = 1.0, 40.0
    nyq = 0.5 * fs
    b,a = butter(3, [low/nyq, high/nyq], btype='band')
    try:
        filtered = filtfilt(b, a, ecg)
    except Exception:
        filtered = ecg
    env = np.abs(filtered)
    # adaptive height: median + 0.5 * std
    med = np.median(env)
    thresh = med + 0.5 * np.std(env)
    distance = int(round(0.18 * fs))
    peaks, props = find_peaks(env, height=thresh, distance=distance)
    return peaks

# Replace or wrap your existing detect_r_peaks call at top-level usage:
def detect_r_peaks_with_fallback(ecg, fs=2000, debug=False):
    # normalize
    ecg_norm = _robust_normalize(ecg)
    # call the original Pan-Tompkins function (the one you already have in file).
    try:
        r = detect_r_peaks(ecg_norm, fs=fs, debug=debug)
    except Exception:
        r = np.array([], dtype=int)
    # if too few peaks, try fallback
    if r is None or len(r) < 2:
        fb = _fallback_qrs_detector(ecg_norm, fs)
        if len(fb) >= 2:
            # refine with ecg peaks (choose maxima in small windows)
            refined = []
            for p in fb:
                lo = max(0, p - int(0.02*fs))
                hi = min(len(ecg_norm)-1, p + int(0.02*fs))
                sub = ecg_norm[lo:hi+1]
                if sub.size == 0: continue
                arg = int(np.argmax(np.abs(sub)))
                refined.append(lo + arg)
            r = np.unique(np.array(refined, dtype=int))
    # final: if still too few, return empty array
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
    Pan-Tompkins style QRS detector.

    Parameters:
        ecg: 1D ECG signal array
        fs: sampling frequency [Hz]
        low_hz/high_hz: bandpass cutoffs [Hz]
        integration_ms: moving window integration length [ms]
        refractory_ms: minimum distance between accepted R peaks [ms]
        search_back_factor: factor to declare a missed-beat gap (default 1.66)
        debug: print debugging info

    Returns:
        numpy array of R-peak indices (samples)
    """
    ecg = np.asarray(ecg).ravel()
    N = ecg.size
    if N == 0:
        return np.array([], dtype=int)

    # 1) bandpass filter
    try:
        ecg_f = _butter_bandpass(ecg, fs, low_hz, high_hz)
    except Exception:
        # fallback: no filtering
        ecg_f = ecg.copy()

    # 2) derivative
    deriv = _five_point_derivative(ecg_f, fs)

    # 3) squaring
    squared = deriv ** 2

    # 4) moving window integration
    mwi = _moving_window_integration(squared, fs, integration_ms)

    # gather candidate peaks from integrated signal
    min_dist = max(1, int(round(refractory_ms / 1000.0 * fs)))
    cand_idxs, _ = find_peaks(mwi, distance=min_dist)

    if cand_idxs.size == 0:
        if debug:
            print("[PT] No MWI peaks found; returning empty array")
        return np.array([], dtype=int)

    cand_vals_i = mwi[cand_idxs]
    cand_vals_f = np.abs(ecg_f[cand_idxs])

    # Initialization using first segment (first n peaks or first 2 seconds)
    # Choose initial candidates from earliest region to compute starting SPKI/NPKI/SPKF/NPKF
    n_init = min(8, cand_idxs.size)
    init_idxs = cand_idxs[:n_init]
    init_i = mwi[init_idxs]
    init_f = np.abs(ecg_f[init_idxs])

    # initial signal peak estimates: take the maximum observed
    SPKI = float(np.max(init_i))
    NPKI = float(np.median(init_i) * 0.5) if init_i.size > 0 else SPKI * 0.125
    SPKF = float(np.max(init_f))
    NPKF = float(np.median(init_f) * 0.5) if init_f.size > 0 else SPKF * 0.125

    THRESHOLD_I = NPKI + 0.25 * (SPKI - NPKI)
    THRESHOLD_F = NPKF + 0.25 * (SPKF - NPKF)
    THRESHOLD_I2 = 0.5 * THRESHOLD_I  # lower threshold for search-back
    THRESHOLD_F2 = 0.5 * THRESHOLD_F

    detected = []
    detected_i_vals = []
    detected_f_vals = []
    rr_intervals = []

    last_accepted = -np.inf

    for idx_cand in cand_idxs:
        val_i = float(mwi[idx_cand])
        val_f = float(np.abs(ecg_f[idx_cand]))

        # enforce refractory
        if detected and (idx_cand - detected[-1]) < min_dist:
            # update noise estimates
            NPKI = 0.125 * val_i + 0.875 * NPKI
            NPKF = 0.125 * val_f + 0.875 * NPKF
            THRESHOLD_I = NPKI + 0.25 * (SPKI - NPKI)
            THRESHOLD_F = NPKF + 0.25 * (SPKF - NPKF)
            THRESHOLD_I2 = 0.5 * THRESHOLD_I
            THRESHOLD_F2 = 0.5 * THRESHOLD_F
            continue

        accept = False
        threshold_used = None

        # primary condition
        if (val_i >= THRESHOLD_I) or (val_f >= THRESHOLD_F):
            accept = True
            threshold_used = 'primary'
        elif (val_i >= THRESHOLD_I2) or (val_f >= THRESHOLD_F2):
            accept = True
            threshold_used = 'secondary'

        if accept:
            # refine to ECG max around candidate
            refined = _refine_to_ecg_peak(ecg, idx_cand, fs, rad_ms=30.0)
            # T-wave discrimination if too close (< 360 ms)
            is_t_wave = False
            if detected:
                prev = detected[-1]
                dt = (refined - prev) / float(fs)
                if dt < 0.36:
                    # compare slopes: if current slope << previous, consider T-wave
                    curr_slope = _max_slope_around(ecg, refined, fs)
                    prev_slope = _max_slope_around(ecg, prev, fs)
                    if prev_slope > 0 and curr_slope < 0.5 * prev_slope:
                        is_t_wave = True

            if is_t_wave:
                # treat as noise and update noise peaks
                NPKI = 0.125 * val_i + 0.875 * NPKI
                NPKF = 0.125 * val_f + 0.875 * NPKF
            else:
                # accept as R-peak if not violating refractory relative to last accepted
                if not detected or (refined - detected[-1]) >= min_dist:
                    detected.append(int(refined))
                    detected_i_vals.append(val_i)
                    detected_f_vals.append(val_f)

                    # update signal estimates
                    if threshold_used == 'primary':
                        SPKI = 0.125 * val_i + 0.875 * SPKI
                        SPKF = 0.125 * val_f + 0.875 * SPKF
                    else:
                        # weaker detection (secondary) updates signal estimates slightly faster
                        SPKI = 0.25 * val_i + 0.75 * SPKI
                        SPKF = 0.25 * val_f + 0.75 * SPKF

                    # update rr lists
                    if len(detected) >= 2:
                        rr = detected[-1] - detected[-2]
                        rr_intervals.append(rr)
                else:
                    # too close after refinement -> treat as noise
                    NPKI = 0.125 * val_i + 0.875 * NPKI
                    NPKF = 0.125 * val_f + 0.875 * NPKF
        else:
            # noise candidate
            NPKI = 0.125 * val_i + 0.875 * NPKI
            NPKF = 0.125 * val_f + 0.875 * NPKF

        # recompute thresholds
        THRESHOLD_I = NPKI + 0.25 * (SPKI - NPKI)
        THRESHOLD_F = NPKF + 0.25 * (SPKF - NPKF)
        THRESHOLD_I2 = 0.5 * THRESHOLD_I
        THRESHOLD_F2 = 0.5 * THRESHOLD_F

    detected = np.array(detected, dtype=int)

    # Search-back for missed beats: if gap > search_back_factor*RR_avg, look for missed peaks in MWI
    if detected.size >= 2:
        rr_ms = np.diff(detected)  # sample counts
        if rr_ms.size > 0:
            # compute RR average (use mean of last up to 8 intervals)
            RRavg = int(np.mean(rr_ms[-8:]))
            if RRavg > 0:
                RRmiss = int(round(search_back_factor * RRavg))
                # scan between detected beats for gaps
                new_found = []
                for i in range(len(detected) - 1):
                    a = detected[i]; b = detected[i+1]
                    gap = b - a
                    if gap > RRmiss:
                        lo = a + 1; hi = b - 1
                        if lo >= hi: continue
                        # find max mwi peak in the gap
                        local_idx_rel = int(np.argmax(mwi[lo:hi+1]))
                        local_idx = lo + local_idx_rel
                        if mwi[local_idx] >= THRESHOLD_I2:
                            refined = _refine_to_ecg_peak(ecg, local_idx, fs, rad_ms=30.0)
                            new_found.append(int(refined))
                            # update SPKI/SPKF faster
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
                # pick index with largest absolute ECG amplitude
                vals = [abs(ecg[idx]) for idx in block]
                keep = block[int(np.argmax(vals))]
                final.append(keep)
            i = j
        detected = np.array(final, dtype=int)

    detected.sort()
    if debug:
        print(f"[Pan-Tompkins] final detected beats: {len(detected)}")

    return detected

def pt_pipeline(ecg: np.ndarray, fs: int = 2000, low_hz: float = 5.0, high_hz: float = 15.0, integration_ms: float = 150.0):
    """
    Return intermediate arrays used by Pan-Tompkins:
      {
        'ecg_raw', 'ecg_filtered', 'deriv', 'squared', 'mwi', 'cand_idxs'
      }
    """
    ecg = np.asarray(ecg).ravel()
    # bandpass
    nyq = 0.5 * fs
    try:
        b, a = butter(3, [low_hz/nyq, high_hz/nyq], btype='band')
        ecg_f = filtfilt(b, a, ecg)
    except Exception:
        ecg_f = ecg.copy()
    # five-point derivative (same kernel used inside)
    kernel = np.array([-1., -2., 0., 2., 1.])
    deriv = np.convolve(ecg_f, kernel, mode='same') * (fs / 8.0)
    squared = deriv ** 2
    win = max(1, int(round(integration_ms / 1000.0 * fs)))
    mwi = np.convolve(squared, np.ones(win) / float(win), mode='same')
    # candidate peaks on MWI (distance = refractory_ms default 200ms -> 0.2s)
    distance = int(round(0.18 * fs))
    cand_idxs, _ = find_peaks(mwi, distance=distance)
    return {
        'ecg_raw': ecg,
        'ecg_filtered': ecg_f,
        'deriv': deriv,
        'squared': squared,
        'mwi': mwi,
        'cand_idxs': cand_idxs
    }

def plot_pt_pipeline(ecg: np.ndarray, fs: int = 2000, r_peaks: np.ndarray = None, figsize=(12, 9), show=True):
    """
    Plot Pan-Tompkins pipeline: raw ECG (+detected R), filtered, derivative+squared, MWI (+candidate peaks).
    If r_peaks provided, they are plotted on the raw ECG axis.
    Returns matplotlib.Figure.
    """
    data = pt_pipeline(ecg, fs)
    t = np.arange(len(ecg)) / float(fs)
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    # 1) Raw ECG
    axes[0].plot(t, data['ecg_raw'])
    if r_peaks is not None and len(r_peaks) > 0:
        axes[0].scatter(r_peaks / float(fs), data['ecg_raw'][r_peaks], c='r', s=12, zorder=3, label='R-peaks')
        axes[0].legend(fontsize='small')
    axes[0].set_ylabel("ECG (raw)")

    # 2) Filtered ECG
    axes[1].plot(t, data['ecg_filtered'])
    axes[1].set_ylabel("Filtered")

    # 3) Derivative & squared
    axes[2].plot(t, data['deriv'], label='derivative')
    axes[2].plot(t, data['squared'], label='squared', alpha=0.7)
    axes[2].legend(fontsize='small')
    axes[2].set_ylabel("Derivative / Squared")

    # 4) Moving window integration (MWI) and candidate peaks
    axes[3].plot(t, data['mwi'], label='MWI (windowed energy)')
    axes[3].scatter(data['cand_idxs'] / float(fs), data['mwi'][data['cand_idxs']], c='orange', s=10, label='candidates')
    axes[3].set_ylabel("MWI")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend(fontsize='small')

    plt.tight_layout()
    if show:
        plt.show()
    return fig