"""
loader.py: Image loading and preprocessing utilities.

This module is the entry point for all images entering the pipeline.
It handles loading, colour conversion, and normalisation.
"""

import numpy as np
import matplotlib.pyplot as plt


def load_image(filepath: str) -> np.ndarray:
    """Load an image from disk and return as a float32 NumPy array.

    Validates that the file exists and has a supported image extension before loading. Warns the user if the image is very large, since convolution on large images takes significant time.

    Supported formats: PNG, JPG, JPEG, BMP, TIFF, WEBP

    Args:
        filepath: Path to the image file.

    Returns:
        Image array of shape (H, W, C) with values in [0, 1].

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the file extension is not a supported image format.
    """
    import os

    # check the file actually exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Image file not found: '{filepath}'\n"
            f"Make sure the file path is correct and the file exists."
        )

    # check the file extension is a supported image format
    supported = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in supported:
        raise ValueError(
            f"Unsupported file format: '{ext}'\n"
            f"Supported formats: {', '.join(sorted(supported))}"
        )

    # load the image
    img = plt.imread(filepath)

    # warn if the image is very large — convolution will be slow
    total_pixels = img.shape[0] * img.shape[1]
    if total_pixels > 4_000_000:
        print(
            f"  Warning: large image ({img.shape[1]}w x {img.shape[0]}h = "
            f"{total_pixels:,} pixels). Convolution may take several minutes."
        )

    # convert from unit8 (0-255) to float32 (0-1) if needed
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    return img


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale using luminosity weights.

    Uses ITU-R BT.601 weights: Y = 0.299R + 0.587G + 0.114B
    These weights reflect human perceptual sensitivity to each colour channel: we are most sensitive to green, least to blue.

    Args:
        image: RGB array of shape (H, W, 3).

    Returns:
        Grayscale array of shape (H, W) with values in [0, 1].
    """
    weights = np.array([0.299, 0.587, 0.114])
    return np.dot(image[..., :3], weights).astype(np.float32)


def normalise(image: np.ndarray) -> np.ndarray:
    """Normalise an array to the range [0, 1].

    Stretches pixel values so the darkest pixel becomes 0 and the brightest pixel becomes 1.

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