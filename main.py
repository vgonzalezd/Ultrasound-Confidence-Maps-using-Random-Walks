import argparse

import scipy.io
from visualization_utils import confidence_plotter, show

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--backend",
        type=str,
        default="numpy",
        help="Backend to use. Can be 'numpy' or 'torch'",
    )

    argparser.add_argument(
        "--precision",
        type=str,
        default="float64",
        help="Precision to use. Can be 'float16', 'float32' or 'float64'",
    )

    # Import confidence map function from the selected backend
    if argparser.parse_args().backend == "numpy":
        from confidence_map_numpy.confidence_map import confidence_map
    else:
        # Give error message if the backend is not supported
        raise NotImplementedError(
            f"The backend \"{argparser.parse_args().backend}\" is not supported."
        )

    # Load neck data and call confidence estimation for B-mode with default parameters
    img = scipy.io.loadmat("data/neck.mat")["img"]
    map_ = confidence_map(img, alpha=2.0, beta=90, gamma=0.03)
    confidence_plotter(img, map_)


    # Load femur data and call confidence estimation for B-mode with default parameters
    img = scipy.io.loadmat("data/femur.mat")["img"]
    map_ = confidence_map(img, alpha=2.0, beta=90, gamma=0.06)
    confidence_plotter(img, map_)

    show()