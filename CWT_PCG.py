# CWT_PCG.py
"""
CWT module with multiple backends and explicit control of scales/translations.
Patched: Pascal backend returns power (squared magnitude) and scalogram is
normalized to [0..1] to make threshold ratios consistent across backends.

Public API:
    compute_cwt(signal, fs, fmin=20, fmax=500, n_freqs=120, backend='auto', **kwargs)
    compute_threshold_and_cogs(scalogram, freqs, times, s1=0.6, s2=0.1, ...)
"""
from typing import Tuple, Optional, Dict, Any
import numpy as np

# try to import pywt/scipy when available
try:
    import pywt  # type: ignore
    _HAS_PYWT = True
except Exception:
    _HAS_PYWT = False

try:
    import scipy.signal as _scipy_signal  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

DEFAULT_PASCAL_A0 = 0.0019
DEFAULT_PASCAL_A_STEP = 0.00031
DEFAULT_PASCAL_F0 = 0.849

__all__ = ["compute_cwt", "compute_cwt_pascal", "compute_threshold_and_cogs"]

def compute_cwt(signal: np.ndarray,
                fs: int,
                fmin: float = 20.0,
                fmax: float = 500.0,
                n_freqs: int = 120,
                backend: str = "auto",
                **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    sig = np.asarray(signal).ravel()
    n = sig.size
    if n == 0:
        return np.zeros((0,0)), np.array([]), np.array([]), "empty"

    backend_try = backend.lower() if backend != "auto" else None
    if backend_try is None:
        if _HAS_PYWT:
            backend_try = "pywt"
        elif _HAS_SCIPY:
            backend_try = "scipy"
        else:
            backend_try = "pascal"

    if backend_try == "pywt":
        if not _HAS_PYWT:
            backend_try = "scipy" if _HAS_SCIPY else "pascal"
        else:
            try:
                wavelet = kwargs.get("wavelet", "morl")
                central = pywt.central_frequency(wavelet)
                freqs = np.linspace(fmin, fmax, n_freqs)
                scales = (central * float(fs)) / freqs
                coeffs, _ = pywt.cwt(sig, scales, wavelet, sampling_period=1.0/float(fs))
                power = (np.abs(coeffs) ** 2)
                times = np.arange(n) / float(fs)
                # normalize to [0..1]
                maxv = np.nanmax(power) if power.size else 0.0
                if maxv > 0:
                    power = power / (maxv + 1e-12)
                return power, freqs, times, "pywt-morl"
            except Exception:
                backend_try = "scipy"

    if backend_try == "scipy":
        if not _HAS_SCIPY:
            backend_try = "spectrogram"
        else:
            try:
                from scipy.signal import cwt, morlet2  # type: ignore
                widths = kwargs.get("widths", np.linspace(1.0, 64.0, n_freqs))
                coeffs = cwt(sig, morlet2, widths, w=5.0)
                approx_freqs = (float(fs) / (widths + 1e-12))
                freqs_target = np.linspace(min(approx_freqs), max(approx_freqs), n_freqs)
                from scipy.interpolate import interp1d  # type: ignore
                interp = interp1d(approx_freqs[::-1], coeffs[::-1, :], axis=0, kind='linear', bounds_error=False, fill_value=0.0)
                coeffs_res = interp(freqs_target)
                power = (np.abs(coeffs_res) ** 2)
                times = np.arange(n) / float(fs)
                maxv = np.nanmax(power) if power.size else 0.0
                if maxv > 0:
                    power = power / (maxv + 1e-12)
                return power, freqs_target, times, "scipy-cwt-morlet2"
            except Exception:
                backend_try = "spectrogram"

    if backend_try == "spectrogram":
        try:
            from scipy.signal import spectrogram  # type: ignore
            nperseg = min(1024, max(128, n // 8))
            noverlap = int(0.75 * nperseg)
            nfft = max(256, 2 ** int(np.ceil(np.log2(nperseg))))
            f, t_spec, Sxx = spectrogram(sig, fs=fs, window='hann',
                                         nperseg=nperseg, noverlap=noverlap,
                                         nfft=nfft, scaling='spectrum', mode='magnitude')
            mask = (f >= fmin) & (f <= fmax)
            if not mask.any():
                return np.zeros((0,0)), np.array([]), np.array([]), "spectrogram-empty-band"
            f_sel = f[mask]
            P_sel = (np.abs(Sxx[mask, :]) ** 2)
            from scipy.interpolate import interp1d  # type: ignore
            freqs_out = np.linspace(f_sel[0], f_sel[-1], n_freqs)
            interp = interp1d(f_sel, P_sel, axis=0, kind='linear', fill_value='extrapolate')
            scalogram = interp(freqs_out)
            maxv = np.nanmax(scalogram) if scalogram.size else 0.0
            if maxv > 0:
                scalogram = scalogram / (maxv + 1e-12)
            return np.abs(scalogram), freqs_out, t_spec, "spectrogram-fallback"
        except Exception:
            return np.zeros((0,0)), np.array([]), np.array([]), "error-no-backend"

    if backend_try == "pascal":
        row_count = int(kwargs.get("row_count", n_freqs))
        col_count = int(kwargs.get("col_count", min(400, max(16, n))))
        a0 = float(kwargs.get("a0", DEFAULT_PASCAL_A0))
        a_step = float(kwargs.get("a_step", DEFAULT_PASCAL_A_STEP))
        f0 = float(kwargs.get("f0", DEFAULT_PASCAL_F0))
        scales = kwargs.get("scales", None)
        freqs_target = kwargs.get("freqs_target", None)
        times = kwargs.get("times", None)
        scal, freqs_out, times_out = compute_cwt_pascal(sig, fs,
                                                        row_count=row_count,
                                                        col_count=col_count,
                                                        a0=a0, a_step=a_step, f0=f0,
                                                        scales=scales,
                                                        freqs_target=freqs_target,
                                                        times=times,
                                                        fmin=fmin, fmax=fmax)
        # ensure normalized
        maxv = np.nanmax(scal) if scal.size else 0.0
        if maxv > 0:
            scal = scal / (maxv + 1e-12)
        return scal, freqs_out, times_out, "pascal-morlett"

    return np.zeros((0,0)), np.array([]), np.array([]), f"unknown-backend:{backend_try}"


def compute_cwt_pascal(signal: np.ndarray,
                       fs: int,
                       row_count: int = 300,
                       col_count: int = 400,
                       a0: float = DEFAULT_PASCAL_A0,
                       a_step: float = DEFAULT_PASCAL_A_STEP,
                       f0: float = DEFAULT_PASCAL_F0,
                       scales: Optional[np.ndarray] = None,
                       freqs_target: Optional[np.ndarray] = None,
                       times: Optional[np.ndarray] = None,
                       fmin: Optional[float] = None,
                       fmax: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(signal, dtype=float).ravel()
    n = x.size
    if n == 0:
        return np.zeros((0,0)), np.array([]), np.array([])

    dt = 1.0 / float(fs)
    total_dur = (n - 1) * dt

    if times is None:
        times_out = np.linspace(0.0, total_dur, num=col_count)
    else:
        times = np.asarray(times, dtype=float).ravel()
        times_out = times.copy()
        col_count = times_out.size

    if scales is not None:
        a_arr_full = np.asarray(scales, dtype=float).ravel()
    elif freqs_target is not None:
        freqs_target = np.asarray(freqs_target, dtype=float).ravel()
        a_arr_full = (f0 / freqs_target)
    else:
        a_arr_full = a0 + np.arange(row_count, dtype=float) * a_step

    freqs_all = (f0 / a_arr_full)
    if (fmin is not None) and (fmax is not None):
        sel_mask = (freqs_all >= fmin) & (freqs_all <= fmax)
        if not sel_mask.any():
            sel_mask = np.ones_like(freqs_all, dtype=bool)
    else:
        sel_mask = np.ones_like(freqs_all, dtype=bool)

    sel_indices = np.nonzero(sel_mask)[0]
    out_rows = sel_indices.size
    if out_rows == 0:
        return np.zeros((0,0)), np.array([]), np.array([])

    scal = np.zeros((out_rows, col_count), dtype=float)
    t_samples = np.arange(n, dtype=float) * dt

    pref_const_base = 1.0 / (np.pi ** 0.25)
    w0 = 2.0 * np.pi * f0

    out_i = 0
    for idx in sel_indices:
        a = float(a_arr_full[idx])
        pref = (1.0 / np.sqrt(a)) * pref_const_base
        b = times_out[:, None]
        tmat = t_samples[None, :]
        tau = (tmat - b) / a
        env = np.exp(-0.5 * (tau ** 2))
        cos_term = np.cos(w0 * tau)
        sin_term = np.sin(w0 * tau)
        kern_re = pref * env * cos_term
        kern_im = pref * env * (-sin_term)
        cwtre_col = kern_re.dot(x)
        cwtim_col = kern_im.dot(x)
        # use power (squared magnitude) to be consistent with pywt's abs(coeffs)**2
        power_col = (cwtre_col ** 2) + (cwtim_col ** 2)
        scal[out_i, :] = power_col
        out_i += 1

    freqs_out = freqs_all[sel_indices]
    return scal, freqs_out, times_out


def compute_threshold_and_cogs(scalogram: np.ndarray,
                               freqs: np.ndarray,
                               times: np.ndarray,
                               s1: float = 0.6,
                               s2: float = 0.10,
                               min_area: Optional[int] = None,
                               keep_top: int = 3) -> Dict[str, Any]:
    """
    Returns dict with 'S1_mask','S2_mask','S1_cog','S2_cog'
    """
    try:
        from Threshold_Plot_CoG import threshold_mask, compute_cog  # type: ignore
    except Exception:
        def threshold_mask_local(scal, ratio, min_area_local=3, keep_top_local=3):
            if scal is None or scal.size == 0:
                return np.zeros_like(scal, dtype=bool)
            peak = float(np.nanmax(scal))
            if peak <= 0:
                return np.zeros_like(scal, dtype=bool)
            thr = ratio * peak
            mask = (scal >= thr)
            return mask

        def compute_cog_local(scal, freqs_arr, times_arr, mask=None):
            if scal is None or scal.size == 0:
                return None
            if mask is None:
                mask = scal > 0
            E = scal * mask
            if not E.any():
                return None
            F, T = np.meshgrid(freqs_arr, times_arr, indexing='ij')
            e = E.flatten()
            t_cog = float((e * T.flatten()).sum() / (e.sum() + 1e-12))
            f_cog = float((e * F.flatten()).sum() / (e.sum() + 1e-12))
            return (t_cog, f_cog)

        threshold_mask = threshold_mask_local
        compute_cog = compute_cog_local

    mask1 = threshold_mask(scalogram, s1, min_area=min_area, keep_top=keep_top)
    mask2 = threshold_mask(scalogram, s2, min_area=min_area, keep_top=keep_top)
    cog1 = compute_cog(scalogram * mask1, freqs, times, mask=mask1)
    cog2 = compute_cog(scalogram * mask2, freqs, times, mask=mask2)
    return {'S1_mask': mask1, 'S2_mask': mask2, 'S1_cog': cog1, 'S2_cog': cog2}
