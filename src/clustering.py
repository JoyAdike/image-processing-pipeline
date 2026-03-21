"""
clustering.py: K-means clustering algorithm from scratch.

This module is the organiser of the pipeline.
It groups pixels by brightness similarity into k clusters, simplifying the image into distinct regions.

K-means is a foundational algorithm in image segmentation, the same mathematical principle used to isolate regions of interest in medical imaging and facial recognition systems.

Implemented using NumPy only.
"""

import numpy as np


def kmeans(
        data: np.ndarray,
        k: int,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        random_seed: int = 42) -> tuple:
    """
    Run k-means clustering on a 1D data array.

    K-means groups data points into k clusters by repeatedly:
        1. Assigning each point to the nearest centroid
        2. Recalculating each centroid as the mean of its cluster
    This continues until the centroids stop moving significantly, known as convergence.

    Real-world analogy: sorting people into k groups by height.
    Each person joins the group whose representative height is closest to their own. Representatives are recalculated each round until nobody needs to move groups anymore.

    Args:
        data: 1D array of values to cluster (e.g. pixel brightness).
        k: Number of clusters to form.
        max_iterations: Maximum number of update rounds.
        tolerance: Convergence threshold - stop when centroids move less than this amount between rounds.
        random_seed: Controls random initialisation for reproducibility.

    Returns:
        Tuple of (labels, centroids).
        labels: Array of shape (N,) — cluster index for each point.
        centroids: Array of shape (k,) — final centroid values.
    """
    # Set random seed so results are reproducible every run
    rng = np.random.default_rng(random_seed)

    n_samples = len(data)

    # Step 1: Initialise: pick k random data points as starting centroids
    # Like randomly picking k people to be the first table representatives
    indices = rng.choice(n_samples, size=k, replace=False)
    centroids = data[indices].copy()

    labels = np.zeros(n_samples, dtype=int)

    for iteration in range(max_iterations):

        # Step 2: Assign: each point joins the nearest centroid
        # Calculate distance from every point to every centroid
        # then assign each point to the closest one
        distances = np.abs(data[:, np.newaxis] - centroids)
        new_labels = np.argmin(distances, axis=1)

        # Step 3: Update: recalculate each centroid as the mean
        # of all points currently assigned to it
        new_centroids = np.array([
            data[new_labels == j].mean() if np.any(new_labels == j)
            else centroids[j]
            for j in range(k)
        ])

        # Step 4: Check convergence: how far did the centroids move?
        # If the maximum movement is below the tolerance, we stop
        movement = np.max(np.abs(new_centroids - centroids))
        labels = new_labels
        centroids = new_centroids

        if movement < tolerance:
            print(f"  K-means converged after {iteration + 1} iterations.")
            break

    return labels, centroids


def segment_image(image: np.ndarray, k: int) -> np.ndarray:
    """
    Segment a grayscale image into k regions using k-means.

    Each pixel is assigned to one of k clusters based on its brightness value. Every pixel is then replaced by its cluster's centroid value — producing a simplified image with exactly k distinct brightness levels.

    Args:
        image: Grayscale image array of shape (H, W) with values in [0, 1].
        k: Number of segments — how many distinct brightness regions to create.

    Returns:
        Segmented image of shape (H, W) where each pixel is replaced by its cluster centroid value.
    """
    # Flatten the 2D image into a 1D array of pixel values
    flat = image.ravel()

    # Run k-means on the flat pixel array
    labels, centroids = kmeans(flat, k=k)

    # Replace each pixel with its cluster's centroid value
    segmented = centroids[labels].reshape(image.shape)

    return segmented.astype(np.float32)