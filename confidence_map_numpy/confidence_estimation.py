import numpy as np
from scipy.sparse.linalg import spsolve

from .confidence_laplacian import confidence_laplacian


def confidence_estimation(A, seeds, labels, beta, gamma):
    """Compute confidence map

    Args:
        A (np.ndarray): Processed image
        seeds (np.ndarray): Seeds for the random walks framework
        labels (np.ndarray): Labels for the random walks framework
        beta: Random walks parameter
        gamma: Horizontal penalty factor

    Returns:
        map: confidence map
    """

    # Index matrix with boundary padding
    G = np.arange(1, A.shape[0] * A.shape[1] + 1).reshape(A.shape[1], A.shape[0]).T
    pad = 1

    G = np.pad(G, (pad, pad), "constant", constant_values=(0, 0))
    B = np.pad(A, (pad, pad), "constant", constant_values=(0, 0))

    # Laplacian
    D = confidence_laplacian(G, B, beta, gamma)

    # Select marked columns from Laplacian to create L_M and B^T
    B = D[:, seeds]

    # Select marked nodes to create B^T
    N = np.sum(G > 0)
    i_U = np.arange(N)
    i_U[seeds.astype(int)] = 0
    i_U = np.where(i_U > 0)[0]  # Index of unmarked nodes
    B = B[i_U, :]

    # Remove marked nodes from Laplacian by deleting rows and cols
    keep_indices = np.setdiff1d(np.arange(D.shape[0]), seeds)
    D = D[keep_indices, :][:, keep_indices]

    # Adjust labels
    label_adjust = np.min(labels, axis=0, keepdims=True)
    labels = labels - label_adjust + 1  # labels > 0

    # Find number of labels (K)
    labels_present = np.unique(labels)
    number_labels = labels_present.shape[0]

    # Define M matrix
    M = np.zeros((seeds.shape[0], number_labels), dtype=np.float64)
    for k in range(number_labels):
        M[:, k] = labels == labels_present[k]

    # Right-handside (-B^T*M)
    rhs = -B @ M  # type: ignore

    # Solve system
    if number_labels == 2:
        x = spsolve(D, rhs[:, 0])
        x = np.vstack((x, 1.0 - x)).T
    else:
        x = spsolve(D, rhs)

    # Prepare output
    probabilities = np.zeros((N, number_labels), dtype=np.float64)  # type: ignore
    for k in range(number_labels):
        # Probabilities for unmarked nodes
        probabilities[i_U, k] = x[:, k]
        # Max probability for marked node of each label
        probabilities[seeds[labels == k].astype(int), k] = 1.0

    # Final reshape with same size as input image (no padding)
    probabilities = probabilities.reshape(
        (A.shape[1], A.shape[0], number_labels)
    ).transpose((1, 0, 2))

    # reshape((A.shape[0], A.shape[1], number_labels))

    return probabilities
