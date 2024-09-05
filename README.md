# SAR imagery processing, download and water extent predictions from DeepAqua

Code originally developed by melqkiades/deep-wetlands

This repository contains scripts for processing SAR (Synthetic Aperture Radar) imagery, including:

1. **Google Earth Engine (GEE) Export**: Exports SAR imagery from Google Earth Engine.2
2. **SAR Image Processing**: Uses a U-Net model to predict water masks from SAR images.


## Overview

### 1. SAR imagery processing and download

This script exports SAR images from Google Earth Engine based on the defined parameters.

**Features:**
- Export individual SAR images for a specific date range.
- Create and export monthly composite SAR images.

### 2. Water extent prediction

This script processes SAR imagery using a U-Net model to generate water masks and export the results.

**Features:**
- Read SAR images and prepare them for processing.
- Predict water masks using a trained U-Net model.
- Export prediction results as GeoTIFFs.
- Calculate and export area estimates for the detected water bodies.

## Prerequisites

### For download_sar.py
- Google Earth Engine account
- Python 3.x
- Required Python packages: `geetools`, `geopandas`, `ee`, `eeconvert`, `unidecode`

### For predict_water.py
- Python 3.x
- Required Python packages: `torch`, `rasterio`, `numpy`, `matplotlib`, `PIL`, `tqdm`, `geopandas`, `shapely`, `pandas`



## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
