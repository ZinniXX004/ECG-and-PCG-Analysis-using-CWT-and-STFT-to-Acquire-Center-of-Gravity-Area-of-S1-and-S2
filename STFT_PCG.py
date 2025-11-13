# STFT_PCG.py
# Simple STFT implementation using only NumPy (no SciPy dependency).
# Returns a time-frequency magnitude (or power) matrix compatible with downstream code.
#
# API:
#   compute_stft(x, fs=2000, nperseg=None, noverlap=None, nfft=None, window='hann',
#                scaling='density', mode='magnitude')
# Returns:
#   Sxx: ndarray (n_freqs, n_times)  -- magnitude or power spectrogram
#   freqs: ndarray (n_freqs,)        -- frequency bin centers in Hz (ascending)
#   times: ndarray (n_times,)        -- times of frame centers in seconds
#   method: str                      -- textual method description
#
# Notes:
# - This implementation intentionally keeps behavior simple and explicit so it's easy to inspect.
# - Uses np.fft.rfft -> returns only non-negative frequencies (shape: nfft//2 + 1).
# - If the signal is shorter than nperseg, it is zero-padded to produce one frame.
# - Window accepts common names ('hann','hamming','rect','box') or an array-like of length nperseg.
# - mode: 'magnitude' (abs), 'power' (abs**2). scaling parameter is present for API compatibility but
#   only influences simple normalization (not full PSD scaling like scipy.spectrogram).
#
# Author: ChatGPT (adapted for your project)
import numpy as np

def _get_window_array(window, nperseg):
    """Return a 1D window array of length nperseg.
    Supported string names: 'hann', 'hamming', 'rect', 'box', 'blackman'
    If window is array-like and matches length, it's returned as float array.
    """
    if window is None:
        return np.ones(nperseg, dtype=float)
    if isinstance(window, str):
        w = window.lower()
        if 'hann' in w:
            return np.hanning(nperseg).astype(float)
        if 'hamming' in w:
            return np.hamming(nperseg).astype(float)
        if 'blackman' in w:
            return np.blackman(nperseg).astype(float)
        if w in ('rect', 'box', 'ones', 'rectangular'):
            return np.ones(nperseg, dtype=float)
        # fallback to hann
        return np.hanning(nperseg).astype(float)
    # array-like
    arr = np.asarray(window, dtype=float)
    if arr.ndim != 1:
        raise ValueError("window must be 1D array or a supported string name")
    if arr.size != nperseg:
        # try to stretch/truncate: simplest is to resample via interpolation
        # but that's usually not desired; better to raise to make user explicit
        raise ValueError(f"window length {arr.size} does not match nperseg {nperseg}")
    return arr

def _next_pow2(x):
    """Return next power of two >= x (used optionally if desired)."""
    return 1 << (int(x) - 1).bit_length() if x > 1 else 1

def compute_stft(x, fs=2000, nperseg=None, noverlap=None, nfft=None, window='hann', scaling='density', mode='magnitude'):
    """
    Compute a simple STFT spectrogram using NumPy only.

    Parameters
    ----------
    x : array-like
        1D input signal.
    fs : int
        Sampling frequency (Hz).
    nperseg : int or None
        Window length (samples). If None, chosen as min(256, max(64, len(x)//8)).
    noverlap : int or None
        Overlap (samples). If None, set to nperseg // 2.
    nfft : int or None
        FFT length. If None, chosen as max(256, nperseg).
    window : str or array-like
        Window description (supported strings: 'hann','hamming','rect','blackman') or array of length nperseg.
    scaling : str
        'density' or 'spectrum' (kept for API compatibility; only simple normalization applied).
    mode : str
        'magnitude' (default) returns |STFT|, 'power' returns |STFT|^2.

    Returns
    -------
    Sxx : ndarray, shape (n_freqs, n_times)
    freqs : ndarray, shape (n_freqs,)
    times : ndarray, shape (n_times,)
    method : str
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size

    if N == 0:
        return np.zeros((0, 0)), np.zeros((0,)), np.zeros((0,)), "STFT (numpy)"

    # sensible defaults
    if nperseg is None:
        nperseg = min(256, max(64, max(1, N // 8)))
    if nfft is None:
        nfft = max(256, nperseg)
    if noverlap is None:
        noverlap = int(nperseg // 2)
    if noverlap >= nperseg:
        noverlap = nperseg // 2

    nperseg = int(max(1, nperseg))
    nfft = int(max(1, nfft))
    noverlap = int(max(0, noverlap))
    hop = nperseg - noverlap
    if hop <= 0:
        hop = 1

    # window
    try:
        win = _get_window_array(window, nperseg)
    except Exception as e:
        raise ValueError(f"Invalid window: {e}")

    # number of frames (include last partial frame by padding)
    if N <= nperseg:
        n_frames = 1
    else:
        n_frames = int(np.ceil((N - noverlap) / float(hop)))

    pad_len = (n_frames - 1) * hop + nperseg
    if pad_len > N:
        x = np.concatenate([x, np.zeros(pad_len - N, dtype=float)])
    elif pad_len < N:
        # should not happen, but safe-guard
        x = x[:pad_len]

    frames = []
    times = []
    for i in range(0, pad_len - nperseg + 1, hop):
        seg = x[i:i + nperseg] * win
        X = np.fft.rfft(seg, n=nfft)
        mag = np.abs(X)
        frames.append(mag)
        # center time of the window
        center = (i + (nperseg / 2.0)) / float(fs)
        times.append(center)

    if len(frames) == 0:
        # fallback: empty
        freqs = np.fft.rfftfreq(nfft, d=1.0 / float(fs))
        return np.zeros((freqs.size, 0)), freqs, np.array([], dtype=float), "STFT (numpy)"

    S = np.column_stack(frames)  # shape: (n_freqs, n_times)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / float(fs))
    times = np.asarray(times, dtype=float)

    # apply simple scaling if requested
    if mode is not None and isinstance(mode, str) and mode.lower().startswith('pow'):
        S = S ** 2
    else:
        # magnitude (default): keep as |X|
        pass

    # optional simple normalization: divide by window energy to keep level stable across windows
    # (this is not the same as scipy's 'density' PSD scaling, but helps usability)
    win_energy = np.sum(win ** 2)
    if win_energy > 0:
        S = S / float(max(1e-12, win_energy))

    method = "STFT (numpy)"
    return np.asarray(S, dtype=float), np.asarray(freqs, dtype=float), np.asarray(times, dtype=float), method


# If run as script, simple self-test (quiet by default).
if __name__ == "__main__":
    # small smoke test
    fs = 1000
    t = np.arange(0, 1.0, 1.0/fs)
    x = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t)
    Sxx, freqs, times, method = compute_stft(x, fs=fs, nperseg=256, noverlap=128, nfft=512, window='hann')
    print("STFT shapes:", Sxx.shape, freqs.shape, times.shape, "method:", method)
