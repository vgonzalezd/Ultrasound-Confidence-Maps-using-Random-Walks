from typing import Tuple

import numpy as np
from scipy.signal import hilbert

from .attenuation_weighting import attenuation_weighting
from .confidence_estimation import confidence_estimation

# Machine epsilon
eps = np.finfo(np.float64).eps


def sub2ind(size: Tuple[int], rows: np.ndarray, cols: np.ndarray):
    """Converts row and column subscripts into linear indices,
    basically the copy of the MATLAB function of the same name.
    https://www.mathworks.com/help/matlab/ref/sub2ind.html

    This function is Pythonic so the indices start at 0.

    Args:
        size Tuple[int]: Size of the matrix
        rows (np.ndarray): Row indices
        cols (np.ndarray): Column indices

    Returns:
        indices (np.ndarray): 1-D array of linear indices
    """
    indices = rows + cols * size[0]
    return indices


def confidence_map(data: np.ndarray, alpha=2.0, beta=90, gamma=0.05, mode="B"):
    """Compute the confidence map

    Args:
        data (np.ndarray): RF ultrasound data (one scanline per column)
        mode: 'RF' or 'B' mode data
        alpha, beta, gamma: See Medical Image Analysis reference

    Returns:
        map (np.ndarray): Confidence map
    """

    print("Preparing confidence estimation...")

    # Normalize data
    data = data.astype(np.float64)
    data = (data - np.min(data)) / ((np.max(data) - np.min(data)) + eps)

    if mode == "RF":
        # MATLAB hilbert applies the Hilbert transform to columns
        data = np.abs(hilbert(data, axis=0)).astype(np.float64)  # type: ignore

    # Seeds and labels (boundary conditions)
    seeds = np.array([], dtype=np.float64)
    labels = np.array([], dtype=np.float64)

    sc = np.arange(data.shape[1], dtype=np.float64)  # All columns

    # SOURCE ELEMENTS - 1st matrix row
    sr_up = np.zeros_like(sc)

    seed = sub2ind(data.shape, sr_up, sc).astype(np.float64)
    seed = np.unique(seed)
    seeds = np.concatenate((seeds, seed))

    # Label 1
    label = np.ones_like(seed)
    labels = np.concatenate((labels, label))

    # SINK ELEMENTS - last image row
    sr_down = np.ones_like(sc) * (data.shape[0] - 1)
    seed = sub2ind(data.shape, sr_down, sc).astype(np.float64)
    seed = np.unique(seed)
    seeds = np.concatenate((seeds, seed))

    # Label 2
    label = np.ones_like(seed) * 2
    labels = np.concatenate((labels, label))

    # Attenuation with Beer-Lambert
    W = attenuation_weighting(data, alpha)

    print("Solving confidence estimation problem, please wait...")

    # Apply weighting directly to image
    # Same as applying it individually during the formation of the Laplacian
    data = data * W

    # Find condidence values
    map = confidence_estimation(data, seeds, labels, beta, gamma)

    # Only keep probabilities for virtual source notes.
    map = map[:, :, 0]

    return map
