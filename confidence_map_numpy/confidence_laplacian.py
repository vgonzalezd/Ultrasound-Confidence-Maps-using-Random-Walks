import numpy as np

from scipy.sparse import csr_matrix

# MATLAB eps
eps = 2.2204e-16


def confidence_laplacian(P, A, beta, gamma):
    """Compute 6-Connected Laplacian for confidence estimation problem

    Args:
        P (np.ndarray): TODO
        A (np.ndarray): TODO

    Returns:
        TODO
    """

    m, n = P.shape

    P = P.T.flatten()
    A = A.T.flatten()

    p = np.where(P > 0)[0]

    i = P[p] - 1  # Index vector
    j = P[p] - 1  # Index vector
    s = np.zeros_like(p)  # Entries vector, initially for diagonal

    vl = 0  # Vertical edges length

    for iter_idx, k in enumerate(
        [
            -1,
            1,  # Vertical edges
            m - 1,
            m + 1,
            -m - 1,
            -m + 1,  # Diagonal edges
            m,
            -m,  # Horizontal edges
        ]
    ):

        Q = P[p + k]

        q = np.where(Q > 0)[0]

        ii = P[p[q]] - 1
        i = np.concatenate((i, ii))
        jj = Q[q] - 1
        j = np.concatenate((j, jj))
        W = np.abs(A[p[ii]] - A[p[jj]])  # Intensity derived weight
        s = np.concatenate((s, W))

        if iter_idx == 1:
            vl = s.shape[0]  # Vertical edges length

    # Normalize weights
    s = (s - np.min(s)) / (np.max(s) - np.min(s) + eps)

    # Horizontal penalty
    s[vl:] += gamma

    # Normalize differences
    s = (s - np.min(s)) / (np.max(s) - np.min(s) + eps)

    # Gaussian weighting function
    EPSILON = 10e-6
    s = -((np.exp(-beta * s)) + EPSILON)

    # Create Laplacian, diagonal missing
    L = csr_matrix((s, (i, j)))

    # Reset diagonal weights to zero for summing
    # up the weighted edge degree in the next step
    L.setdiag(0)

    # Weighted edge degree
    D = np.abs(L.sum(axis=0).A)[0]

    # Finalize Laplacian by completing the diagonal
    L.setdiag(D)

    return L
