"""
histogram.py: Histogram computation and contrast equalisation.

This module is the contrast specialist of the pipeline. It analyses the distribution of pixel brightness values and
redistributes them to improve image contrast and reveal hidden detail, particularly in dark or washed-out images.
"""

import numpy as np


def compute_histogram(
        image: np.ndarray,
        bins: int = 256) -> tuple:
    """Compute the pixel intensity histogram of a grayscale image.

    A histogram counts how many pixels have each brightness value. It shows the distribution of light and dark across the image.

    Args:
        image: Grayscale image with values in [0, 1].
        bins: Number of histogram bins (default 256).

    Returns:
        Tuple of (counts, bin_centres).
        counts: Number of pixels in each brightness bin.
        bin_centres: The brightness value at the centre of each bin.
    """
    # Flatten the 2D image into a 1D list of all pixel values
    counts, edges = np.histogram(image.ravel(), bins=bins, range=(0.0, 1.0))

    # Calculate the centre point of each bin
    bin_centres = (edges[:-1] + edges[1:]) / 2

    return counts, bin_centres


def equalise_histogram(
        image: np.ndarray,
        bins: int = 256) -> np.ndarray:
    """Equalise the histogram of a grayscale image to improve contrast.

    Histogram equalisation redistributes pixel brightness values so they are spread evenly across the full range [0, 1].

    How it works:
        1. Compute the histogram: count pixels at each brightness level
        2. Compute the CDF: for each brightness, what fraction of all pixels are darker than this value?
        3. Use the CDF as a lookup table: each pixel's original brightness maps to a new brightness value based on the CDF
        4. The result is an image with balanced contrast

    The CDF is the key mathematical tool here. It answers:
    "What percentage of pixels are darker than this brightness value?"
    A dark image has a CDF that rises steeply at low values.
    After equalisation, the CDF becomes a straight diagonal line, meaning brightness is distributed evenly.

    Args:
        image: Grayscale image with values in [0, 1].
        bins: Number of histogram bins.

    Returns:
        Equalised image with values in [0, 1], same shape as input.
    """
    # Step 1: compute the histogram
    counts, _ = compute_histogram(image, bins)

    # Step 2: compute the CDF as a running cumulative sum
    cdf = np.cumsum(counts)

    # Step 3: normalise the CDF to the range [0, 1]
    # This converts raw pixel counts into fractions of the total
    cdf = cdf / cdf[-1]

    # Step 4: map each pixel through the CDF lookup table
    # First convert pixel values to bin indices
    bin_indices = np.floor(image * (bins - 1)).astype(int)
    bin_indices = np.clip(bin_indices, 0, bins - 1)

    # Look up the new brightness value for each pixel
    equalised = cdf[bin_indices].astype(np.float32)

    return equalised