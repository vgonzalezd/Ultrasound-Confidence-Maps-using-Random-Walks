import argparse
import os
import time

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform

from confidence_map_numpy import ConfidenceMap as ConfidenceMap_numpy
from confidence_map_cupy import ConfidenceMap as ConfidenceMap_cupy
from confidence_map_oct import ConfidenceMap as ConfidenceMap_oct

def save_results(img, map_, output_path):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title("Input")

    plt.subplot(1, 2, 2)
    plt.imshow(map_, cmap="gray")
    plt.axis("off")
    plt.title("Confidence map")

    plt.savefig(output_path)
    plt.close()

def main(args : argparse.Namespace) -> None:

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
            f'The backend "{argparser.parse_args().backend}" is not supported.'
        )

    # Check if the precision is supported
    if args.precision not in ["float32", "float64"]:
        raise NotImplementedError(
            f'The precision "{argparser.parse_args().precision}" is not supported.'
        )

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    img = nib.load(args.input)
    img_data = img.get_fdata()

    #label = nib.load(args.label)
    #label_data = label.get_fdata()

    # Create confidence map object
    cm = ConfidenceMap(args.precision, alpha=2.0, beta=90.0, gamma=0.03)

    total_processing_time = 0
    for i in range(img_data.shape[2]):
        print(f"Processing slice {i}...")

        start_time = time.time()
        map_ = cm(img_data[..., i])
        total_processing_time += time.time() - start_time
        
        # Save results
        save_results(img_data[..., i], map_, os.path.join(args.output, f"{i}.png"))

    print(f"Total processing time: {total_processing_time} seconds")
    print(f"Average processing time per slice: {total_processing_time / img_data.shape[2]} seconds")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--backend",
        type=str,
        default="numpy",
        help="Backend to use. Can be 'numpy' or 'cupy'",
    )
    argparser.add_argument(
        "--input",
        type=str,
        default="../data/30.nii",
        help="Input file",
    )
    argparser.add_argument(
        "--label",
        type=str,
        default="../data/30-labels.nii",
        help="Label file",
    )
    argparser.add_argument(
        "--precision",
        type=str,
        default="float64",
        help="Precision to use. Can be 'float32' or 'float64'",
    )
    argparser.add_argument(
        "--output",
        type=str,
        default="../nii_test/",
        help="Output directory",
    )

    args = argparser.parse_args()

    main(args)