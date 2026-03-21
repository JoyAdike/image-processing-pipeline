"""
test_filters.py : Unit tests for the image processing pipeline.

These tests verify that the core mathematical properties of each module are correct. They run automatically and confirm the pipeline behaves as expected.

Run with:
    python -m pytest tests/test_filters.py -v
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.filters import make_gaussian_kernel, convolve2d, sobel_edge_detection
from src.loader import normalise
from src.histogram import compute_histogram, equalise_histogram
from src.clustering import kmeans


def test_gaussian_kernel_sums_to_one():
    """A Gaussian kernel must sum to 1.0 to preserve image brightness. If the kernel does not sum to 1, blurring would make the image brighter or darker, which is incorrect behaviour.
    """
    kernel = make_gaussian_kernel(size=5, sigma=1.0)
    assert abs(kernel.sum() - 1.0) < 1e-6, (
        f"Gaussian kernel sum should be 1.0 but got {kernel.sum()}"
    )


def test_gaussian_kernel_shape():
    """The kernel shape must match the requested size."""
    kernel = make_gaussian_kernel(size=7, sigma=1.5)
    assert kernel.shape == (7, 7), (
        f"Expected shape (7, 7) but got {kernel.shape}"
    )


def test_gaussian_kernel_centre_is_highest():
    """
    The centre pixel must have the highest weight.
    The Gaussian bell curve peaks at the centre — the pixel being processed gets the most weight, neighbours get less.
    """
    kernel = make_gaussian_kernel(size=5, sigma=1.0)
    centre = kernel[2, 2]
    assert centre == kernel.max(), (
        "Centre of Gaussian kernel should have the highest weight"
    )


def test_flat_image_produces_zero_edges():
    """
    A completely flat image has no brightness changes in its interior.
    The Sobel operator should return zero edges in the interior pixels.
    Note: border pixels may show small artifacts due to zero-padding,this is expected behaviour where the padding creates an artificial brightness change at the image boundary.
    """
    flat_image = np.ones((50, 50), dtype=np.float32) * 0.5
    magnitude, _, _ = sobel_edge_detection(flat_image)

    # Check only interior pixels — exclude 2-pixel border where
    # zero-padding creates artificial brightness changes
    interior = magnitude[2:-2, 2:-2]
    assert interior.max() < 1e-6, (
        "Interior of flat image should have zero edge magnitude"
    )


def test_normalise_range():
    """After normalisation, min must be 0.0 and max must be 1.0."""
    image = np.array([[10, 50], [100, 200]], dtype=np.float32)
    normalised = normalise(image)
    assert abs(normalised.min() - 0.0) < 1e-6, "Min should be 0.0"
    assert abs(normalised.max() - 1.0) < 1e-6, "Max should be 1.0"


def test_histogram_counts_all_pixels():
    """
    The histogram must count every pixel exactly once.
    Total counts across all bins must equal total pixels in image.
    """
    image = np.random.rand(100, 100).astype(np.float32)
    counts, _ = compute_histogram(image)
    assert counts.sum() == image.size, (
        f"Histogram should count {image.size} pixels "
        f"but counted {counts.sum()}"
    )


def test_equalised_image_range():
    """Equalised image must stay within [0, 1] range."""
    image = np.random.rand(100, 100).astype(np.float32)
    equalised = equalise_histogram(image)
    assert equalised.min() >= 0.0, "Equalised image min should be >= 0"
    assert equalised.max() <= 1.0, "Equalised image max should be <= 1"


def test_kmeans_returns_correct_number_of_clusters():
    """K-means must return exactly k unique centroid values."""
    data = np.random.rand(500).astype(np.float32)
    labels, centroids = kmeans(data, k=4)
    assert len(centroids) == 4, (
        f"Expected 4 centroids but got {len(centroids)}"
    )


def test_kmeans_labels_match_data_length():
    """Every data point must receive a cluster label."""
    data = np.random.rand(300).astype(np.float32)
    labels, _ = kmeans(data, k=3)
    assert len(labels) == len(data), (
        f"Expected {len(data)} labels but got {len(labels)}"
    )


if __name__ == "__main__":
    tests = [
        test_gaussian_kernel_sums_to_one,
        test_gaussian_kernel_shape,
        test_gaussian_kernel_centre_is_highest,
        test_flat_image_produces_zero_edges,
        test_normalise_range,
        test_histogram_counts_all_pixels,
        test_equalised_image_range,
        test_kmeans_returns_correct_number_of_clusters,
        test_kmeans_labels_match_data_length,
    ]

    print("\nRunning tests...\n")
    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"  PASS — {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL — {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests.")