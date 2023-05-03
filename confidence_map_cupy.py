from typing import Literal, Tuple

import cupy as cp
import numpy as np

from cupyx.scipy.sparse.linalg import cg, spilu, LinearOperator
from cupyx.scipy.sparse import csc_matrix

from scipy.linalg import hilbert


CONJUGATE_GRADIENT_MAX_ITERATIONS = 200
CONJUGATE_GRADIENT_TOLERANCE = 1e-6


class ConfidenceMap:
    """Confidence map computation class for RF ultrasound data"""

    def __init__(
        self,
        precision: Literal["float16", "float32", "float64"] = "float32",
        alpha: float = 2.0,
        beta: float = 90.0,
        gamma: float = 0.05,
        mode: Literal["RF", "B"] = "B",
    ):
        """Compute the confidence map

        Args:
            precision: Precision to use. Can be 'float16', 'float32' or 'float64'
            alpha (float, optional): Alpha parameter. Defaults to 2.0.
            beta (float, optional): Beta parameter. Defaults to 90.0.
            gamma (float, optional): Gamma parameter. Defaults to 0.05.
            mode (str, optional): 'RF' or 'B' mode data. Defaults to 'B'.
        """

        # The hyperparameters for confidence map estimation
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mode = mode

        # The precision to use for all computations
        self.precision = precision
        self.eps = cp.finfo(self.precision).eps

    def normalize(self, inp: cp.ndarray) -> cp.ndarray:
        """Normalize an array to [0, 1]"""
        return (inp - cp.min(inp)) / (cp.ptp(inp) + self.eps)

    def attenuation_weighting(self, A: cp.ndarray, alpha: float) -> cp.ndarray:
        """Compute attenuation weighting

        Args:
            A (cp.ndarray): Image
            alpha: Attenuation coefficient (see publication)

        Returns:
            W (cp.ndarray): Weighting expressing depth-dependent attenuation
        """

        # Create depth vector and repeat it for each column
        Dw = cp.linspace(0, 1, A.shape[0], dtype=self.precision)
        Dw = cp.tile(Dw.reshape(-1, 1), (1, A.shape[1])) # type: ignore

        W = 1.0 - cp.exp(-alpha * Dw)  # Compute exp inline

        return W

    def confidence_laplacian(
        self, P: cp.ndarray, A: cp.ndarray, beta: float, gamma: float
    ) -> csc_matrix:
        """Compute 6-Connected Laplacian for confidence estimation problem

        Args:
            P (cp.ndarray): The index matrix of the image with boundary padding.
            A (cp.ndarray): The padded image.
            beta (float): Random walks parameter that defines the sensitivity of the Gaussian weighting function.
            gamma (float): Horizontal penalty factor that adjusts the weight of horizontal edges in the Laplacian.

        Returns:
            L (csc_matrix): The 6-connected Laplacian matrix used for confidence map estimation.
        """

        m, _ = P.shape

        P = P.T.flatten()
        A = A.T.flatten()

        p = cp.where(P > 0)[0]

        i = P[p] - 1  # Index vector
        j = P[p] - 1  # Index vector
        # Entries vector, initially for diagonal
        s = cp.zeros_like(p, dtype=self.precision)

        vl = 0  # Vertical edges length

        edge_templates = [
            -1,  # Vertical edges
            1,
            m - 1,  # Diagonal edges
            m + 1,
            -m - 1,
            -m + 1,
            m,  # Horizontal edges
            -m,
        ]

        for iter_idx, k in enumerate(edge_templates):

            Q = P[p + k]

            q = cp.where(Q > 0)[0]

            ii = P[p[q]] - 1
            i = cp.concatenate((i, ii))
            jj = Q[q] - 1
            j = cp.concatenate((j, jj))
            W = cp.abs(A[p[ii]] - A[p[jj]])  # Intensity derived weight
            s = cp.concatenate((s, W))

            if iter_idx == 1:
                vl = s.shape[0]  # Vertical edges length

        # Normalize weights
        s = self.normalize(s)

        # Horizontal penalty
        s[vl:] += gamma

        # Normalize differences
        s = self.normalize(s)

        # Gaussian weighting function
        s = -(
            (cp.exp(-beta * s, dtype=self.precision)) + 1.0e-6
        )  # --> This epsilon changes results drastically default: 1.e-6

        # Create Laplacian, diagonal missing
        L = csc_matrix((s, (i, j)))

        # Diagonal indices
        d = cp.arange(0, L.shape[0], dtype=cp.int32)

        # Reset diagonal weights to zero for summing
        # up the weighted edge degree in the next step
        L[d, d] = 0

        # Weighted edge degree
        D = cp.abs(L.sum(axis=0))[0]

        # Finalize Laplacian by completing the diagonal
        L[d, d] = D

        return L

    def confidence_estimation(self, A, seeds, labels, beta, gamma):
        """Compute confidence map

        Args:
            A (cp.ndarray): Processed image.
            seeds (cp.ndarray): Seeds for the random walks framework. These are indices of the source and sink nodes.
            labels (cp.ndarray): Labels for the random walks framework. These represent the classes or groups of the seeds.
            beta: Random walks parameter that defines the sensitivity of the Gaussian weighting function.
            gamma: Horizontal penalty factor that adjusts the weight of horizontal edges in the Laplacian.

        Returns:
            map: Confidence map which shows the probability of each pixel belonging to the source or sink group.
        """

        # Index matrix with boundary padding
        G = cp.arange(1, A.shape[0] * A.shape[1] + 1).reshape(A.shape[1], A.shape[0]).T
        pad = 1

        G = cp.pad(G, (pad, pad), "constant", constant_values=(0, 0))
        B = cp.pad(A, (pad, pad), "constant", constant_values=(0, 0))

        # Laplacian
        D = self.confidence_laplacian(G, B, beta, gamma)

        # Select marked columns from Laplacian to create L_M and B^T
        B = D[:, seeds]

        # Select marked nodes to create B^T
        N = cp.sum(G > 0).item()
        i_U = np.setdiff1d(np.arange(N), cp.asnumpy(seeds).astype(int))  # Index of unmarked nodes
        i_U = cp.asarray(i_U)
        B = B[i_U, :]

        # Remove marked nodes from Laplacian by deleting rows and cols
        keep_indices = np.setdiff1d(np.arange(D.shape[0]), cp.asnumpy(seeds))
        keep_indices = cp.asarray(keep_indices)
        D = csc_matrix(D[keep_indices, :][:, keep_indices])

        # Define M matrix
        M = cp.zeros((seeds.shape[0], 1), dtype=self.precision)
        M[:, 0] = labels == 1

        # Right-handside (-B^T*M)
        rhs = -B @ M  # type: ignore

        # Compute an incomplete LU decomposition for use as a preconditioner
        lu = spilu(D)
        preconditioner_M = LinearOperator(
            D.shape, lu.solve, dtype=self.precision  # type: ignore
        )  # Create a linear operator to use as the preconditioner

        # Solve system
        x = cg(
            D,
            rhs,
            tol=CONJUGATE_GRADIENT_TOLERANCE,
            maxiter=CONJUGATE_GRADIENT_MAX_ITERATIONS,
            M=preconditioner_M,
        )[0]

        # Prepare output
        probabilities = cp.zeros((N,), dtype=self.precision)
        # Probabilities for unmarked nodes
        probabilities[i_U] = x
        # Max probability for marked node
        probabilities[seeds[labels == 1].astype(int)] = 1.0

        # Final reshape with same size as input image (no padding)
        probabilities = probabilities.reshape((A.shape[1], A.shape[0])).T

        return probabilities

    def sub2ind(
        self, size: Tuple[int], rows: cp.ndarray, cols: cp.ndarray
    ) -> cp.ndarray:
        """Converts row and column subscripts into linear indices,
        basically the copy of the MATLAB function of the same name.
        https://www.mathworks.com/help/matlab/ref/sub2ind.html

        This function is Pythonic so the indices start at 0.

        Args:
            size Tuple[int]: Size of the matrix
            rows (cp.ndarray): Row indices
            cols (cp.ndarray): Column indices

        Returns:
            indices (cp.ndarray): 1-D array of linear indices
        """
        indices = rows + cols * size[0]
        return indices

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Compute the confidence map

        Args:
            data (np.ndarray): RF ultrasound data (one scanline per column)

        Returns:
            map (np.ndarray): Confidence map
        """

        print("Preparing confidence estimation...")

        # convert to cupy
        data = cp.array(data)

        # Normalize data
        data = data.astype(self.precision)
        data = self.normalize(data)

        if self.mode == "RF":
            # MATLAB hilbert applies the Hilbert transform to columns

            # convert to numpy
            data = cp.asnumpy(data)

            # Apply Hilbert transform
            data = np.abs(hilbert(data, axis=0)).astype(self.precision)  # type: ignore

            # convert to cupy
            data = cp.array(data)


        # Seeds and labels (boundary conditions)
        seeds = cp.array([], dtype=self.precision)
        labels = cp.array([], dtype=self.precision)

        # Indices for all columns
        sc = cp.arange(data.shape[1], dtype=self.precision)

        # SOURCE ELEMENTS - 1st matrix row
        # Indices for 1st row, it will be broadcasted with sc
        sr_up = cp.array([0])
        seed = self.sub2ind(data.shape, sr_up, sc).astype(self.precision)
        seed = cp.unique(seed)
        seeds = cp.concatenate((seeds, seed))

        # Label 1
        label = cp.ones_like(seed)
        labels = cp.concatenate((labels, label))

        # SINK ELEMENTS - last image row
        sr_down = cp.ones_like(sc) * (data.shape[0] - 1)
        seed = self.sub2ind(data.shape, sr_down, sc).astype(self.precision)
        seed = cp.unique(seed)
        seeds = cp.concatenate((seeds, seed))

        # Label 2
        label = cp.ones_like(seed) * 2
        labels = cp.concatenate((labels, label))

        # Attenuation with Beer-Lambert
        W = self.attenuation_weighting(data, self.alpha)

        print("Solving confidence estimation problem, please wait...")

        # Apply weighting directly to image
        # Same as applying it individually during the formation of the
        # Laplacian
        data = data * W

        # Find condidence values
        map_ = self.confidence_estimation(data, seeds, labels, self.beta, self.gamma)
        map_ = cp.asnumpy(map_)

        return map_
