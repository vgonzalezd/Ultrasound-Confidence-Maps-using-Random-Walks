import matplotlib.pyplot as plt
import numpy as np


def confidence_plotter(img, map_):
    """Utility function to plot confidence map and image side by side

    Args:
        img (np.ndarray): Image
        map_ (np.ndarray): Confidence map

    Returns:
        None
    """
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(map_, cmap="gray")
    plt.axis("off")


def save_as_npy(map_, filename):
    """Utility function to save confidence map as .npy file

    Args:
        map_ (np.ndarray): Confidence map
        filename (str): Filename

    Returns:
        None
    """
    np.save(filename, map_)

def show():
    """Utility function to show all plots"""
    plt.show()
