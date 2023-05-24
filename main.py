import argparse

import scipy.io
from visualization_utils import confidence_plotter, show, save_as_npy

from confidence_map_numpy import ConfidenceMap as ConfidenceMap_numpy
from confidence_map_cupy import ConfidenceMap as ConfidenceMap_cupy
from confidence_map_oct import ConfidenceMap as ConfidenceMap_oct

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--backend",
        type=str,
        default="octave",
        help="Backend to use. Can be 'numpy' or 'cupy' or 'octave'",
    )

    argparser.add_argument(
        "--precision",
        type=str,
        default="float64",
        help="Precision to use. Can be 'float32' or 'float64'",
    )

    args = argparser.parse_args()

    # Import confidence map function from the selected backend
    if args.backend == "numpy":
        ConfidenceMap = ConfidenceMap_numpy
    elif args.backend == "cupy":
        ConfidenceMap = ConfidenceMap_cupy
    elif args.backend == "octave":
        ConfidenceMap = ConfidenceMap_oct
    else:
        # Give error message if the backend is not supported
        raise NotImplementedError(
            f'The backend "{args.backend}" is not supported.'
        )

    # Check if the precision is supported
    if args.precision not in ["float32", "float64"]:
        raise NotImplementedError(
            f'The precision "{args.precision}" is not supported.'
        )

    # Load neck data and call confidence estimation for B-mode with default parameters
    img = scipy.io.loadmat("data/neck.mat")["img"]
    cm = ConfidenceMap(
        args.precision, alpha=2.0, beta=90.0, gamma=0.03
    )
    map_ = cm(img)
    save_as_npy(map_, "data/neck_result.npy")
    confidence_plotter(img, map_)

    # Load femur data and call confidence estimation for B-mode with default parameters
    img = scipy.io.loadmat("data/femur.mat")["img"]
    cm = ConfidenceMap(
        args.precision, alpha=2.0, beta=90.0, gamma=0.06
    )
    map_ = cm(img)
    save_as_npy(map_, "data/femur_result.npy")

    confidence_plotter(img, map_)

    show()
