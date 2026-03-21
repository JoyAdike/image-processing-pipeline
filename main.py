"""
main.py: Image Processing Pipeline entry point.

This is the studio manager of the pipeline. It receives an image file, calls each specialist module in the correct
order, passes each module's output to the next, and produces a complete visual display of all pipeline stages.

Usage:
    python main.py --image path/to/your/image.jpg

Example:
    python main.py --image test_image.jpg
"""

import argparse
import numpy as np

from src.loader import load_image, to_grayscale, normalise
from src.filters import gaussian_blur, sobel_edge_detection
from src.histogram import equalise_histogram
from src.clustering import segment_image
from src.visualise import (show_pipeline_stages, plot_histogram_comparison, plot_gaussian_curve)


def run_pipeline(image_path: str) -> None:
    """
    Execute the full image processing pipeline on one image.

    Runs all five stages in sequence:
        1. Load and convert to grayscale
        2. Apply Gaussian blur to reduce noise
        3. Detect edges using the Sobel operator
        4. Equalise histogram to improve contrast
        5. Segment into regions using k-means clustering

    Then produces three visualisations:
        - All five pipeline stages side by side
        - Histogram before and after equalisation
        - Gaussian curves for different sigma values

    Args:
        image_path: Path to the input image file.
    """
    print("=" * 50)
    print("  Image Processing Pipeline")
    print("=" * 50)

    # Stage 1 — Load and standardise the image
    print("\n[1/5] Loading image...")
    image = load_image(image_path)
    gray = to_grayscale(image)
    gray = normalise(gray)
    print(f"  Image loaded: {gray.shape[1]}w x {gray.shape[0]}h pixels")

    # Stage 2 — Gaussian blur to reduce noise
    print("\n[2/5] Applying Gaussian blur (size=7, sigma=1.5)...")
    blurred = gaussian_blur(gray, size=7, sigma=1.5)
    print("  Blur complete.")

    # Stage 3 — Sobel edge detection
    print("\n[3/5] Detecting edges with Sobel operator...")
    edges, grad_x, grad_y = sobel_edge_detection(blurred)
    edges = normalise(edges)
    print(f"  Edges detected. Strongest edge magnitude: "
          f"{round(float(edges.max()), 4)}")

    # Stage 4 — Histogram equalisation
    print("\n[4/5] Equalising histogram...")
    equalised = equalise_histogram(gray)
    print("  Equalisation complete.")

    # Stage 5 — K-means segmentation
    print("\n[5/5] Segmenting image with k-means (k=4)...")
    segmented = segment_image(gray, k=4)

    # Display all results
    print("\nGenerating visualisations...")
    show_pipeline_stages(gray, blurred, edges, equalised, segmented)
    plot_histogram_comparison(gray, equalised)
    plot_gaussian_curve([0.5, 1.0, 2.0, 4.0])

    print("\n" + "=" * 50)
    print("  Pipeline complete.")
    print("  Three output images saved to your project folder.")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image Processing Pipeline — NHL Stenden CV&DS"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image file (JPG or PNG)"
    )
    args = parser.parse_args()
    run_pipeline(args.image)