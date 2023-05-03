import argparse

import numpy as np
import matplotlib.pyplot as plt


def main():
    """Main function"""

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--name",
        type=str,
        default="neck_result",
        help="Name of the data to be processed. For example, 'neck' or 'femur'.",
    )
    argparser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory where the data is stored.",
    )

    # in CSV format
    matlab_output = np.loadtxt(
        f"{argparser.parse_args().data_dir}/{argparser.parse_args().name}.csv",
        delimiter=",",
    )
    python_output = np.load(
        f"{argparser.parse_args().data_dir}/{argparser.parse_args().name}.npy"
    )

    print(f"Matlab output shape: {matlab_output.shape}")
    print(f"Python output shape: {python_output.shape}")

    print(f"Matlab output type: {matlab_output.dtype}")
    print(f"Python output type: {python_output.dtype}")

    print(f"Matlab output min: {np.min(matlab_output)}")
    print(f"Python output min: {np.min(python_output)}")

    print(f"Matlab output max: {np.max(matlab_output)}")
    print(f"Python output max: {np.max(python_output)}")

    print(f"Matlab output mean: {np.mean(matlab_output)}")
    print(f"Python output mean: {np.mean(python_output)}")

    print(f"Matlab output std: {np.std(matlab_output)}")
    print(f"Python output std: {np.std(python_output)}")

    # MSE
    print(f"MSE: {np.mean((matlab_output - python_output)**2)}")

    # RMSE
    print(f"RMSE: {np.sqrt(np.mean((matlab_output - python_output)**2))}")

    # Display RMSE with colorbar
    plt.figure()
    plt.imshow(np.abs(matlab_output - python_output), cmap="gray")
    plt.colorbar()
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
