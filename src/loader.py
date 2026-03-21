"""
loader.py: Image loading and preprocessing utilities.

This module is the entry point for all images entering the pipeline.
It handles loading, colour conversion, and normalisation.
"""

import numpy as np
import matplotlib.pyplot as plt


def load_image(filepath: str) -> np.ndarray:
    """Load an image from disk and return as a float32 NumPy array.

    The image is loaded using matplotlib and converted to float32
    with pixel values in the range [0, 1].

    Args:
        filepath: Path to the image file (PNG, JPG, etc.)

    Returns:
        Image array of shape (H, W, C) with values in [0, 1].
    """
    img = plt.imread(filepath)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return img


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale using luminosity weights.

    Uses ITU-R BT.601 weights: Y = 0.299R + 0.587G + 0.114B
    These weights reflect human perceptual sensitivity to each
    colour channel: we are most sensitive to green, least to blue.

    Args:
        image: RGB array of shape (H, W, 3).

    Returns:
        Grayscale array of shape (H, W) with values in [0, 1].
    """
    weights = np.array([0.299, 0.587, 0.114])
    return np.dot(image[..., :3], weights).astype(np.float32)


def normalise(image: np.ndarray) -> np.ndarray:
    """Normalise an array to the range [0, 1].

    Stretches pixel values so the darkest pixel becomes 0
    and the brightest pixel becomes 1.

    Args:
        image: Input array of any shape.

    Returns:
        Array with values linearly scaled to [0, 1].
    """
    min_val = image.min()
    max_val = image.max()
    if max_val == min_val:
        return np.zeros_like(image)
    return ((image - min_val) / (max_val - min_val)).astype(np.float32)