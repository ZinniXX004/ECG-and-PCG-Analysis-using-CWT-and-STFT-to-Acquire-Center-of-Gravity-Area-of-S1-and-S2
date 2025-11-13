"""
Segmentation_ECG_to_PCG.py (patched)

Segment PCG using ECG R-peak indices.

Updated:
 - segment_one_cycle: added optional parameters pre_ms/post_ms, enforce_single_cycle,
   allow_partial_edges to avoid capturing adjacent cycles.
 - choose_clean_beat: improved heuristic (envelope-based SNR-like) to pick a "clean" beat index.

Compatibility:
 - Returns & main signatures unchanged; old calls without new params behave as before.
"""

from typing import Tuple, Optional
import numpy as np


def segment_one_cycle(pcg: np.ndarray,
                      r_peaks: np.ndarray,
                      idx: int = 0,
                      pad_ms: float = 50.0,
                      fs: int = 2000,
                      pre_ms: Optional[float] = None,
                      post_ms: Optional[float] = None,
                      enforce_single_cycle: bool = True,
                      allow_partial_edges: bool = True) -> Tuple[np.ndarray, int, int]:
    """
    Extract a single PCG segment corresponding to the RR interval r_peaks[idx] -> r_peaks[idx+1].

    Behaviour (defaults preserve previous behavior):
      - By default a symmetric padding `pad_ms` (ms) is applied before the first R and after the next R.
      - If `pre_ms` or `post_ms` provided, they override the respective side padding.
      - If `enforce_single_cycle` is True (default), the computed start/end are clipped to the midpoints
        between adjacent R-peaks so the returned window does NOT cross into adjacent cycles.
      - If the requested index is near the file edges and `allow_partial_edges` is False, an IndexError
        is raised. If True (default), clipping to 0/len(pcg) is used as necessary.

    Returns:
        (segment_array, start_sample, end_sample)

    Raises:
        IndexError if idx out of range (same as before) or if neighboring midpoints are required but not allowed.
    """
    if idx < 0 or idx >= len(r_peaks) - 1:
        raise IndexError("idx must be within 0..len(r_peaks)-2")

    # determine left/right pad in samples
    if pre_ms is None:
        pre_ms = pad_ms
    if post_ms is None:
        post_ms = pad_ms
    pre_samp = int(round(pre_ms / 1000.0 * fs))
    post_samp = int(round(post_ms / 1000.0 * fs))

    # naive start/end (may be trimmed below)
    start = r_peaks[idx] - pre_samp
    end = r_peaks[idx + 1] + post_samp

    # enforce boundaries
    start = int(max(0, start))
    end = int(min(len(pcg), end))

    # If requested, clip to midpoints between adjacent R-peaks to avoid crossing into neighbor cycles
    if enforce_single_cycle:
        # compute midpoint to previous R (between r_peaks[idx-1] and r_peaks[idx])
        if idx > 0:
            prev_mid = int(round(0.5 * (r_peaks[idx - 1] + r_peaks[idx])))
        else:
            prev_mid = 0
        # compute midpoint to next next R (between r_peaks[idx+1] and r_peaks[idx+2])
        if idx + 2 < len(r_peaks):
            next_mid = int(round(0.5 * (r_peaks[idx + 1] + r_peaks[idx + 2])))
        else:
            next_mid = len(pcg)

        # If user asked to disallow using partial edges and a midpoint is missing near the requested interval,
        # raise an error for clarity; otherwise we clamp to 0/len(pcg).
        if not allow_partial_edges:
            if idx == 0 and prev_mid == 0 and (r_peaks[idx] - pre_samp) < 0:
                raise IndexError("Left-side midpoint unavailable and allow_partial_edges=False")
            if (idx + 2) >= len(r_peaks) and next_mid == len(pcg) and (r_peaks[idx + 1] + post_samp) > len(pcg):
                raise IndexError("Right-side midpoint unavailable and allow_partial_edges=False")

        # clip start/end into exclusive midpoints region so we don't include neighbor cycles
        start = max(start, prev_mid)
        end = min(end, next_mid)

        # ensure not inverted (rare); if inverted, fallback to original padding clipped to signal
        if start >= end:
            # fallback to minimal safe RR window without padding
            start = r_peaks[idx]
            end = r_peaks[idx + 1]
            # still enforce in-bounds
            start = max(0, start); end = min(len(pcg), end)

    # final clipping to array bounds (safety)
    start = int(max(0, min(start, len(pcg))))
    end = int(max(0, min(end, len(pcg))))

    return pcg[start:end].copy(), start, end


def choose_clean_beat(pcg: np.ndarray,
                      r_peaks: np.ndarray,
                      fs: int = 2000,
                      method: str = 'envelope_ratio',
                      envelope_win_ms: float = 50.0,
                      search_pad_ms: float = 30.0) -> int:
    """
    Heuristic to select a 'clean' beat index.

    Parameters:
        pcg: 1D PCG signal array.
        r_peaks: array of R-peak sample indices (must contain at least two).
        fs: sampling frequency (Hz).
        method: currently only 'envelope_ratio' supported (higher is better).
        envelope_win_ms: smoothing window (ms) for envelope used in SNR-like measure.
        search_pad_ms: when measuring per-beat envelope, how much extra padding (ms) around RR interval
                       to include in the metric.

    Returns:
        index (int) of chosen beat (0-based index into r_peaks for start of RR).
        If no valid beats found, returns 0 (backwards-compatible default).

    Behavior:
        'envelope_ratio' computes, for each RR interval, an envelope E (smoothed abs(PCG)) and then computes
        score = max(E_segment) / (median(E_segment) + eps). The beat with largest score is returned.
        This is a simple SNR-like heuristic that tends to prefer beats with distinct S1/S2 components
        and low baseline noise.
    """
    if r_peaks is None or len(r_peaks) < 2:
        return 0
    pcg = np.asarray(pcg).ravel()
    if pcg.size == 0:
        return 0

    # prepare envelope (smoothed absolute)
    env_win = max(1, int(round(envelope_win_ms / 1000.0 * fs)))
    env_kernel = np.ones(env_win) / float(env_win)
    env = np.convolve(np.abs(pcg), env_kernel, mode='same')

    # pad in samples for local measurement
    pad_s = int(round(search_pad_ms / 1000.0 * fs))

    scores = []
    for i in range(len(r_peaks) - 1):
        # obtain candidate segment bounds, but do not apply midpoint clipping here;
        # we only want a local window around the RR interval for scoring.
        start = max(0, r_peaks[i] - pad_s)
        end = min(len(pcg), r_peaks[i + 1] + pad_s)
        if end <= start:
            scores.append(0.0)
            continue
        seg_env = env[start:end]
        if seg_env.size == 0:
            scores.append(0.0)
            continue
        # compute simple ratio: peak / median (robust)
        peak = float(np.max(seg_env))
        med = float(np.median(seg_env)) + 1e-12
        score = peak / med
        scores.append(score)

    scores = np.array(scores, dtype=float)
    if np.all(np.isfinite(scores)) and scores.size > 0 and np.nanmax(scores) > 0:
        best_idx = int(np.nanargmax(scores))
        return best_idx
    else:
        return 0
