"""
Segment PCG using ECG R-peak indices.

Functions:
 - segment_one_cycle(pcg, r_peaks, idx=0, pad_ms=50, fs=2000) -> (segment, start_idx, end_idx)
 - choose_clean_beat(...) simple helper (returns index 0 by default).
"""

from typing import Tuple
import numpy as np


def segment_one_cycle(pcg: np.ndarray,
                      r_peaks: np.ndarray,
                      idx: int = 0,
                      pad_ms: float = 50.0,
                      fs: int = 2000) -> Tuple[np.ndarray, int, int]:
    """
    Extract a single PCG segment corresponding to the RR interval r_peaks[idx] -> r_peaks[idx+1].
    Adds symmetrical padding (pad_ms) to ensure S1/S2 edges are captured.

    Returns (segment_array, start_sample, end_sample)
    """
    if idx < 0 or idx >= len(r_peaks) - 1:
        raise IndexError("idx must be within 0..len(r_peaks)-2")

    start = int(max(0, r_peaks[idx] - int(round(pad_ms / 1000.0 * fs))))
    end = int(min(len(pcg), r_peaks[idx + 1] + int(round(pad_ms / 1000.0 * fs))))
    return pcg[start:end].copy(), start, end


def choose_clean_beat(pcg: np.ndarray, r_peaks: np.ndarray, fs: int = 2000) -> int:
    """
    Heuristic to select a 'clean' beat index. Currently returns 0.
    You can implement SNR or envelope checks to pick a better beat.
    """
    # Placeholder: could compute envelope variance and pick highest SNR beat
    return 0
