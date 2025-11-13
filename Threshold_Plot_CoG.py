# Threshold_Plot_CoG.py
"""
Threshold_Plot_CoG.py (numpy-only)
- Implements threshold_mask and compute_cog without scipy.ndimage.
- Uses simple connected-component labeling (8-connectivity) via flood-fill.
- API compatible with previous version.
"""
import numpy as np

def _ensure_mask_shape(mask, target_shape):
    if mask is None:
        return np.zeros(target_shape, dtype=bool)
    mask = np.asarray(mask)
    if mask.shape == target_shape:
        return mask.astype(bool)
    if mask.T.shape == target_shape:
        return mask.T.astype(bool)
    if mask.ndim == 1:
        if mask.shape[0] == target_shape[1]:
            return np.tile(mask.astype(bool), (target_shape[0], 1))
        if mask.shape[0] == target_shape[0]:
            return np.tile(mask.astype(bool)[:, None], (1, target_shape[1]))
    try:
        m = (mask != 0).astype(bool)
        if m.ndim == 1 and m.shape[0] == target_shape[0]:
            return np.tile(m[:, None], (1, target_shape[1]))
    except Exception:
        pass
    return np.zeros(target_shape, dtype=bool)

# -------------- connected-component labeling (8-connectivity) --------------
def _label_components(bin_mask):
    """
    Label connected components in 2D boolean array using 8-connectivity.
    Returns labeled (same shape, int32), ncomponents (int), and components -> list of (ys,xs) arrays optional.
    """
    mask = np.asarray(bin_mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    label = 0
    # neighbors offsets (8-connectivity)
    neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for i in range(h):
        for j in range(w):
            if not mask[i, j] or labels[i, j] != 0:
                continue
            label += 1
            # flood fill using stack
            stack = [(i, j)]
            labels[i, j] = label
            while stack:
                y, x = stack.pop()
                for dy, dx in neigh:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = label
                        stack.append((ny, nx))
    return labels, label

def threshold_mask(scalogram, thr_ratio, min_area=None, keep_top=3, freqs=None, times=None, time_window=None):
    S = np.asarray(scalogram)
    if S.ndim != 2:
        raise ValueError("scalogram must be 2D")
    n_freqs, n_times = S.shape
    maxS = np.nanmax(S)
    if maxS == 0 or np.isnan(maxS):
        return np.zeros_like(S, dtype=bool)
    global_thr = float(thr_ratio) * float(maxS)
    base_mask = (S >= global_thr)

    # label components
    labeled, ncomp = _label_components(base_mask)
    if ncomp == 0:
        return np.zeros_like(S, dtype=bool)

    # compute areas
    areas = []
    for idx in range(1, ncomp+1):
        ys, xs = np.nonzero(labeled == idx)
        areas.append(len(ys))
    areas = np.array(areas, dtype=int)

    if min_area is None:
        min_area_adaptive = max(3, int(0.005 * S.size))
    else:
        min_area_adaptive = int(min_area)

    keep_mask = np.zeros_like(S, dtype=bool)
    for idx, area in enumerate(areas, start=1):
        if area >= min_area_adaptive:
            keep_mask |= (labeled == idx)
    if not keep_mask.any():
        order = np.argsort(-areas)
        for k in range(min(keep_top, len(order))):
            comp_idx = int(order[k]) + 1
            keep_mask |= (labeled == comp_idx)

    # time window refinement (compute centroid times for components)
    if time_window is not None and times is not None:
        t0, t1 = float(time_window[0]), float(time_window[1])
        comps = np.unique(labeled[keep_mask])
        refined_mask = np.zeros_like(keep_mask, dtype=bool)
        times_arr = np.asarray(times, dtype=float)
        for comp in comps:
            if comp == 0:
                continue
            comp_mask = (labeled == comp)
            if not comp_mask.any():
                continue
            ys, xs = np.nonzero(comp_mask)
            if xs.size == 0:
                continue
            col_mean = float(xs.mean())
            col_idx = int(round(col_mean))
            col_idx = max(0, min(col_idx, len(times_arr)-1))
            t_comp = float(times_arr[col_idx])
            if (t_comp >= t0) and (t_comp <= t1):
                refined_mask |= comp_mask
        return refined_mask.astype(bool)

    return keep_mask.astype(bool)

def compute_cog(scalogram, freqs, times, mask=None):
    S = np.asarray(scalogram, dtype=float)
    if S.ndim != 2:
        raise ValueError("scalogram must be 2D")
    n_freqs, n_times = S.shape
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

    W = np.abs(S) * M
    total = np.nansum(W)
    if not np.isfinite(total) or total <= 0:
        return None

    time_weights = np.nansum(W, axis=0)
    freq_weights = np.nansum(W, axis=1)

    t_cog = float(np.nansum(time_weights * times) / np.nansum(time_weights))
    f_cog = float(np.nansum(freq_weights * freqs) / np.nansum(freq_weights))

    return (t_cog, f_cog)
