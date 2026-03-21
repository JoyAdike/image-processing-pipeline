"""
filters.py: Image convolution and filtering algorithms.

This module is the clarity specialist of the pipeline.
It cleans the image with Gaussian blur and finds edge using the Sobel operator. All algorithms are implemented from scratch using NumPy only, no OpenCV or scikit-image.
"""

import numpy as np


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a 2D convolution kernel to a grayscale image.

    Convolution works by sliding a small grid of numbers (the kernel) across every position in the image. At each position, we multiply the kernel values by the pixel values underneath and sum the results. This sum becomes the new pixel value at that position.

    Zero-padding is applied to the borders so the output image is the same size as the input image.

    Args:
        image: Grayscale image array of shape (H, W).
        kernel: 2D convolution kernel of shape (kH, kW).

    Returns:
        Convolved image of shape (H, W).
    """
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    # Pad the image edges with zeros so border pixels are handled correctly
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    output = np.zeros_like(image)

    # Slide the kernel across every pixel position
    for i in range(image_h):
        for j in range(image_w):
            # Extract the region of the image under the kernel
            region = padded[i:i + kernel_h, j:j + kernel_w]
            # Multiply element-wise and sum, this is the convolution operation
            output[i, j] = np.sum(region * kernel)

    return output


def make_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Create a 2D Gaussian kernel.

    The Gaussian function defines a bell curve shape:
        G(x, y) = exp(-(x^2 + y^2) / (2 * sigma^2))

    The kernel is normalised so all values sum to 1.
    This preserves the overall brightness of the image after blurring, we are redistributing brightness, not adding or removing it.

    Args:
        size: Kernel size, must be an odd number (e.g. 5 for a 5x5 kernel).
        sigma: Standard deviation, controls the width of the bell curve.
               Larger sigma = wider blur. Smaller sigma = subtle blur.

    Returns:
        Normalised 2D Gaussian kernel of shape (size, size).
    """
    centre = size // 2

    # Create a grid of x and y coordinates centred at zero
    x, y = np.mgrid[-centre:centre + 1, -centre:centre + 1]

    # Apply the Gaussian formula to every position in the grid
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Normalise so all values sum to 1, preserves image brightness
    return kernel / kernel.sum()


def gaussian_blur(image: np.ndarray, size: int = 5,
                  sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur to reduce noise in an image.

    Blurring works by replacing each pixel with a weighted average of its neighbours. Nearby pixels get more weight, distant pixels get less, following the Gaussian bell curve distribution.

    We blur before edge detection so noise does not create false edges.

    Args:
        image: Grayscale image array of shape (H, W).
        size: Kernel size, larger means stronger blur (must be odd).
        sigma: Standard deviation of the Gaussian distribution.

    Returns:
        Blurred image of shape (H, W).
    """
    kernel = make_gaussian_kernel(size, sigma)
    return convolve2d(image, kernel)


def sobel_edge_detection(
        image: np.ndarray) -> tuple:
    """Detect edges using the Sobel operator.

    An edge is a place where pixel brightness changes rapidly.
    The Sobel operator measures this rate of change (the gradient) in two directions separately horizontal and vertical, then combines them into an overall edge strength map.

    Sobel kernels:
        Gx detects vertical edges (horizontal brightness changes)
        Gy detects horizontal edges (vertical brightness changes)

    The gradient magnitude combines both:
        magnitude = sqrt(Gx^2 + Gy^2)

    Args:
        image: Grayscale image array of shape (H, W).

    Returns:
        Tuple of (magnitude, gradient_x, gradient_y).
        magnitude: Overall edge strength at every pixel.
        gradient_x: Horizontal gradient (vertical edges).
        gradient_y: Vertical gradient (horizontal edges).
    """
    # Sobel kernel for detecting horizontal changes (vertical edges)
    sobel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float32)

    # Sobel kernel for detecting vertical changes (horizontal edges)
    sobel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]], dtype=np.float32)

    grad_x = convolve2d(image, sobel_x)
    grad_y = convolve2d(image, sobel_y)

    # Combine horizontal and vertical gradients using Pythagoras
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    return magnitude, grad_x, grad_y