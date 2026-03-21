"""
visualise.py : Matplotlib visualisation utilities for the pipeline.

This module is the display artist of the pipeline.
It takes the numerical output of every other module and renders it as visual images and graphs that humans can understand and interpret.

"""

import numpy as np
import matplotlib.pyplot as plt


def show_pipeline_stages(
        original: np.ndarray,
        blurred: np.ndarray,
        edges: np.ndarray,
        equalised: np.ndarray,
        segmented: np.ndarray,
        save_path: str = "pipeline_output.png") -> None:
    """
    Display all five pipeline stages side by side in one figure.

    Creates a gallery-style display showing the effect of each transformation on the original image. Each stage is labelled and displayed in grayscale.

    Args:
        original: Original grayscale image.
        blurred: Gaussian-blurred image.
        edges: Sobel edge magnitude image.
        equalised: Histogram-equalised image.
        segmented: K-means segmented image.
        save_path: File path to save the figure as a PNG.
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    images = [original, blurred, edges, equalised, segmented]
    titles = [
        "1. Original",
        "2. Gaussian Blur",
        "3. Sobel Edges",
        "4. Equalised",
        "5. K-Means (k=4)"
    ]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle(
        "Image Processing Pipeline — All Stages",
        fontsize=13,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Pipeline output saved to: {save_path}")
    plt.show()


def plot_histogram_comparison(
        before: np.ndarray,
        after: np.ndarray,
        save_path: str = "histogram_comparison.png") -> None:
    """
    Plot pixel brightness histograms before and after equalisation.

    Shows two bar charts side by side, the distribution of pixel brightness values before and after histogram equalisation. A dark image will show a histogram skewed to the left before equalisation, and a more balanced distribution after.

    Args:
        before: Original grayscale image.
        after: Histogram-equalised image.
        save_path: File path to save the figure as a PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    labels = ["Before equalisation", "After equalisation"]
    colors = ["steelblue", "seagreen"]

    for ax, img, label, color in zip(axes, [before, after],
                                     labels, colors):
        ax.hist(
            img.ravel(),
            bins=64,
            color=color,
            edgecolor='none',
            alpha=0.85
        )
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel("Pixel brightness (0 = black, 1 = white)")
        ax.set_ylabel("Number of pixels")
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Histogram Equalisation — Brightness Distribution",
        fontsize=13,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Histogram comparison saved to: {save_path}")
    plt.show()


def plot_gaussian_curve(
        sigma_values: list,
        save_path: str = "gaussian_curves.png") -> None:
    """Visualise Gaussian curves for different sigma values.

    Shows how sigma controls the width and spread of the Gaussian bell curve. Useful for understanding how different sigma values produce different amounts of blur.

    Args:
        sigma_values: List of sigma values to compare.
        save_path: File path to save the figure as a PNG.
    """
    x = np.linspace(-5, 5, 300)

    fig, ax = plt.subplots(figsize=(9, 4))

    for sigma in sigma_values:
        # Gaussian formula — the bell curve
        y = np.exp(-x**2 / (2 * sigma**2))
        # Normalise so peak is always 1 for visual comparison
        y = y / y.max()
        ax.plot(x, y, linewidth=2, label=f"σ = {sigma}")

    ax.set_xlabel("Distance from centre pixel")
    ax.set_ylabel("Weight given to neighbour")
    ax.set_title(
        "Gaussian distribution : how sigma controls blur strength",
        fontweight='bold'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Gaussian curves saved to: {save_path}")
    plt.show()