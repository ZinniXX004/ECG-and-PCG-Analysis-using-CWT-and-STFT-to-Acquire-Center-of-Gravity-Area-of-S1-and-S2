"""
Threshold_Plot_CoG.py
- threshold_mask: returns boolean mask of same shape as scalogram (n_freqs, n_times)
  - uses scipy.ndimage.label when available to compute connected components and area filtering
  - fallback: basic global threshold + keep_top largest columns/rows if scipy not available
- compute_cog: center-of-gravity in (time, frequency) units; robust to mask shapes and None inputs
"""

import numpy as np

# prefer scipy.ndimage for labeling and center_of_mass
try:
    from scipy import ndimage as spnd
except Exception:
    spnd = None


def _ensure_mask_shape(mask, target_shape):
    """Ensure mask has same shape as target_shape (n_freqs, n_times). If a transpose mismatch,
    attempt to transpose; else try broadcast/crop/resize conservatively."""
    if mask is None:
        return np.zeros(target_shape, dtype=bool)
    mask = np.asarray(mask)
    if mask.shape == target_shape:
        return mask.astype(bool)
    # if transposed
    if mask.T.shape == target_shape:
        return mask.T.astype(bool)
    # If 1D vector for times or freqs, expand
    if mask.ndim == 1:
        # if length matches times -> repeat rows
        if mask.shape[0] == target_shape[1]:
            return np.tile(mask.astype(bool), (target_shape[0], 1))
        if mask.shape[0] == target_shape[0]:
            return np.tile(mask.astype(bool)[:, None], (1, target_shape[1]))
    # fallback: threshold nonzero -> broadcast if possible, else zeros
    try:
        m = (mask != 0).astype(bool)
        # try to broadcast
        if m.shape[0] == target_shape[0] and m.ndim == 1:
            return np.tile(m[:, None], (1, target_shape[1]))
    except Exception:
        pass
    return np.zeros(target_shape, dtype=bool)


def threshold_mask(scalogram, thr_ratio, min_area=None, keep_top=3):
    """
    Create boolean mask on scalogram using threshold ratio thr_ratio (0..1).
    - scalogram: 2D array (n_freqs, n_times) of power/magnitude (non-negative)
    - thr_ratio: fraction relative to scalogram max (e.g. 0.6 -> keep values >= 0.6*max)
    - min_area: minimum connected-component area in pixels (None => adaptive)
    - keep_top: if no region meets min_area, keep top-N largest components
    Returns: boolean mask with same shape as scalogram.
    """
    S = np.asarray(scalogram)
    if S.ndim != 2:
        raise ValueError("scalogram must be 2D")

    n_freqs, n_times = S.shape
    if np.nanmax(S) == 0 or np.isnan(np.nanmax(S)):
        return np.zeros_like(S, dtype=bool)

    # baseline threshold
    global_thr = float(thr_ratio) * float(np.nanmax(S))
    base_mask = (S >= global_thr)

    # if scipy available, use connected components and area filtering
    if spnd is not None:
        labeled, ncomp = spnd.label(base_mask)
        if ncomp == 0:
            return np.zeros_like(S, dtype=bool)
        # collect areas
        areas = spnd.sum(base_mask, labeled, index=np.arange(1, ncomp+1))
        areas = np.asarray(areas, dtype=float)
        # choose min_area adaptively if None: e.g., 0.5% of total pixels or 3 pixels, whichever larger
        if min_area is None:
            min_area_adaptive = max(3, int(0.005 * S.size))
        else:
            min_area_adaptive = int(min_area)
        keep_mask = np.zeros_like(S, dtype=bool)
        # keep components whose area >= min_area_adaptive
        for idx, area in enumerate(areas, start=1):
            if area >= min_area_adaptive:
                keep_mask |= (labeled == idx)
        if keep_mask.any():
            return keep_mask.astype(bool)
        # otherwise fall back to keeping top-K largest components
        # sort indices by area desc
        order = np.argsort(-areas)
        for k in range(min(keep_top, len(order))):
            comp_idx = int(order[k]) + 1
            keep_mask |= (labeled == comp_idx)
        return keep_mask.astype(bool)

    # if scipy not available -> fallback: morphological-like approach using column/row sums
    # Keep columns where sum of base_mask is large, and rows where base_mask sum is large
    col_sums = np.sum(base_mask, axis=0)
    row_sums = np.sum(base_mask, axis=1)
    # choose thresholds
    col_thr = max(1, int(0.01 * n_freqs))  # at least 1 pixel or 1% of rows
    row_thr = max(1, int(0.01 * n_times))
    strong_cols = col_sums >= col_thr
    strong_rows = row_sums >= row_thr
    fallback_mask = np.zeros_like(S, dtype=bool)
    if strong_cols.any() and strong_rows.any():
        fallback_mask[np.ix_(strong_rows, strong_cols)] = True
        return fallback_mask.astype(bool)
    # last resort: return base_mask
    return base_mask.astype(bool)


def compute_cog(scalogram, freqs, times, mask=None):
    """
    Compute center-of-gravity (CoG) of scalogram in time & frequency units.
    - scalogram: 2D array (n_freqs, n_times)
    - freqs: 1D array length n_freqs (Hz)
    - times: 1D array length n_times (seconds, absolute)
    - mask: boolean array same shape (n_freqs, n_times) or None
    Returns: (t_cog, f_cog) or None if no energy.
    """
    S = np.asarray(scalogram, dtype=float)
    if S.ndim != 2:
        raise ValueError("scalogram must be 2D")
    n_freqs, n_times = S.shape

    # ensure freqs/times are arrays
    if freqs is None or len(freqs) != n_freqs:
        freqs = np.linspace(0.0, 1.0, n_freqs)
    if times is None or len(times) != n_times:
        times = np.linspace(0.0, float(n_times-1), n_times)

    freqs = np.asarray(freqs, dtype=float)
    times = np.asarray(times, dtype=float)

    if mask is None:
        M = np.ones_like(S, dtype=bool)
    else:
        M = _ensure_mask_shape(mask, (n_freqs, n_times))

    # weighted by magnitude (abs)
    W = np.abs(S) * M
    total = np.nansum(W)
    if not np.isfinite(total) or total <= 0:
        return None

    # compute weighted mean indices
    # time: sum over freq axis -> weight each time by sum over freqs
    time_weights = np.nansum(W, axis=0)  # shape (n_times,)
    # freq: sum over time axis -> weight each freq by sum over times
    freq_weights = np.nansum(W, axis=1)  # shape (n_freqs,)

    # compute center positions in physical units
    t_cog = float(np.nansum(time_weights * times) / np.nansum(time_weights))
    f_cog = float(np.nansum(freq_weights * freqs) / np.nansum(freq_weights))

    return (t_cog, f_cog)
