# How to Run the Image Processing Pipeline
## Setup and usage guide for any machine

---

## Requirements

Before you start, make sure you have the following installed:

- [Anaconda](https://www.anaconda.com/download) : for managing the Python environment
- [Git](https://git-scm.com/downloads) : for cloning the repository

No other installations are needed. All Python dependencies are handled by Anaconda.

---

## Step 1 : Clone the repository

Open your terminal (Mac/Linux) or Anaconda Prompt (Windows) and run:

```bash
git clone https://github.com/JoyAdike/image-processing-pipeline.git
cd image-processing-pipeline
```

---

## Step 2 : Create the Anaconda environment

This creates an isolated Python environment with exactly the packages the project needs:

```bash
conda env create -f environment.yml
```

This reads the `environment.yml` file and installs Python 3.11, NumPy and Matplotlib automatically. It takes about a minute.

---

## Step 3 : Activate the environment

**Mac / Linux:**
```bash
conda activate image-pipeline
```

**Windows:**
```bash
conda activate image-pipeline
```

You will see `(image-pipeline)` appear at the start of your terminal prompt. This confirms the environment is active.

---

## Step 4 : Add your image

Copy any JPG or PNG image into the project folder. For example:

**Mac / Linux:**
```bash
cp ~/Downloads/your_photo.jpg .
```

**Windows:**
```
copy C:\Users\YourName\Downloads\your_photo.jpg .
```

Or simply drag and drop the image file into the `image-processing-pipeline` folder using your file browser.

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

---

## Step 5 : Run the pipeline

```bash
python main.py --image your_photo.jpg
```

Replace `your_photo.jpg` with the actual filename of the image you added.

---

## What you will see

The pipeline runs five stages and prints progress to the terminal:

```
==================================================
  Image Processing Pipeline
==================================================

[1/5] Loading image...
  Image loaded: 1440w x 1800h pixels

[2/5] Applying Gaussian blur (size=7, sigma=1.5)...
  Blur complete.

[3/5] Detecting edges with Sobel operator...
  Edges detected. Strongest edge magnitude: 1.0

[4/5] Equalising histogram...
  Equalisation complete.

[5/5] Segmenting image with k-means (k=4)...
  K-means converged after 41 iterations.

Generating visualisations...
```

Three windows will open showing:
1. All five pipeline stages side by side
2. Histogram before and after equalisation
3. Gaussian curves for different sigma values

Three PNG files are also saved to the project folder:
- `pipeline_output.png`
- `histogram_comparison.png`
- `gaussian_curves.png`

---

## Step 6 : Run the tests

To verify all modules are working correctly:

```bash
python tests/test_filters.py
```

Expected output:
```
Running tests...

  PASS — test_gaussian_kernel_sums_to_one
  PASS — test_gaussian_kernel_shape
  PASS — test_gaussian_kernel_centre_is_highest
  PASS — test_flat_image_produces_zero_edges
  PASS — test_normalise_range
  PASS — test_histogram_counts_all_pixels
  PASS — test_equalised_image_range
  PASS — test_kmeans_returns_correct_number_of_clusters
  PASS — test_kmeans_labels_match_data_length

9 passed, 0 failed out of 9 tests.
```

---

## Common issues

**"conda: command not found"**
Anaconda is not installed or not on your PATH. Download and install it from anaconda.com/download, then close and reopen your terminal.

**"FileNotFoundError: Image file not found"**
The image filename you typed does not match the actual file in the folder. Check the spelling and make sure the image is inside the `image-processing-pipeline` folder, not in a subfolder.

**"ValueError: Unsupported file format"**
The file you are trying to use is not a supported image format. Rename it with a `.jpg` or `.png` extension, or use a different image file.

**The pipeline takes a long time**
This is normal for large images. Convolution processes every pixel individually. A 1800×1440 image takes approximately 1-2 minutes on a standard laptop. Smaller images run faster.

**On Windows, use Anaconda Prompt instead of Command Prompt**
The `conda` command only works in Anaconda Prompt on Windows, not in the standard Command Prompt or PowerShell.

---

## Project structure

```
image-processing-pipeline/
├── src/
│   ├── loader.py        # Image loading, validation, grayscale conversion
│   ├── filters.py       # Gaussian blur and Sobel edge detection
│   ├── histogram.py     # Histogram equalisation using CDF
│   ├── clustering.py    # K-means image segmentation
│   └── visualise.py     # Matplotlib output and plotting
├── tests/
│   └── test_filters.py  # Automated tests — 9 passing
├── main.py              # Entry point — run this to start the pipeline
├── environment.yml      # Anaconda environment definition
├── requirements.txt     # Python package requirements
└── README.md
```

---

## Deactivating the environment when finished

```bash
conda deactivate
```