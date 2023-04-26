import numpy as np


def xexp(x, a):
    return np.exp(-a * x)


def attenuation_weighting(A, alpha):
    """Compute attenuation weighting

    Args:
        A (np.ndarray): Image
        alpha: Attenuation coefficient (see publication)

    Returns:
        W (np.ndarray): Weighting expresing depth-dependent attenuation
    """

    Dw = np.arange(A.shape[0])
    Dw = Dw / A.shape[0]
    Dw = np.repeat(Dw.reshape(-1, 1), A.shape[1], axis=1)

    Dw = (Dw - np.min(Dw, axis=0, keepdims=True)) / (
        np.max(Dw, axis=0, keepdims=True) - np.min(Dw, axis=0, keepdims=True)
    )
    W = 1.0 - xexp(Dw, alpha)

    return W
