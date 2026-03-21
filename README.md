# Image Processing Pipeline

An image pre-processing pipeline built using NumPy and Matplotlib.


---

## Why I Built This

I want to develop the technical foundation for a global clinical system that uses facial recognition to identify conscious or unconscious patients and retrieve their medical history, enabling correct emergency treatment regardless of location or language barriers.

---

## What It Does

Takes any image and runs it through five processing stages:

1. **Load and standardise** : converts any image to a normalised grayscale float array
2. **Gaussian blur** : reduces noise using a mathematically precise weighted averaging kernel
3. **Sobel edge detection** : finds boundaries by computing the image gradient in two directions
4. **Histogram equalisation** : improves contrast by redistributing pixel brightness using the CDF
5. **K-means segmentation** : groups pixels into k regions by brightness similarity

---

## Mathematical Concepts Used

| Module | Concepts |
|--------|----------|
| loader.py | Normalisation, linear scaling, matrix operations |
| filters.py | Gaussian distribution, convolution, derivatives, vector norm |
| histogram.py | Probability distribution, CDF, random variables, mean |
| clustering.py | Mean, variance, Euclidean distance, convergence |
| visualise.py | Data visualisation, bell curve, distribution comparison |

---

## Project Structure
```
image-processing-pipeline/
├── src/
│   ├── loader.py        # Image loading and standardisation
│   ├── filters.py       # Gaussian blur and Sobel edges
│   ├── histogram.py     # Histogram equalisation
│   ├── clustering.py    # K-means segmentation
│   └── visualise.py     # Matplotlib visualisations
├── tests/
│   └── test_filters.py  # 9 unit tests — all passing
├── main.py              # Pipeline entry point
├── environment.yml      # Anaconda environment definition
├── requirements.txt     # Python dependencies
└── README.md
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/JoyAdike/image-processing-pipeline.git
cd image-processing-pipeline
```

### 2. Create and activate the Anaconda environment
```bash
conda env create -f environment.yml
conda activate image-pipeline
```

### 3. Run the pipeline
```bash
python main.py --image your_image.jpg
```

### 4. Run the tests
```bash
python tests/test_filters.py
```

---

## Output

Running the pipeline produces three saved images:

- `pipeline_output.png` : all five stages side by side
- `histogram_comparison.png` : brightness distribution before and after equalisation
- `gaussian_curves.png` : Gaussian bell curves for different sigma values

---

## Dependencies

- Python 3.11
- NumPy
- Matplotlib

No other libraries. Every algorithm is implemented from scratch.