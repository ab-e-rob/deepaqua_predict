import os
import cv2
import rasterio.features
from rasterio.windows import Window
import torch
from models.unet import Unet
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import shape, MultiPolygon
import csv
import pandas as pd

### Set parameters
STUDY_AREA = 'Helge'
MODEL_NAME = 'big-2020'
PATCH_SIZE = 64
BULK_EXPORT_DIR = f'sar_imagery/{STUDY_AREA}_sar_export'
RESULTS_DIR = f'prediction_tiffs/{STUDY_AREA}/'
PRETRAINED_MODEL_DIR = f'pretrained_models/{MODEL_NAME}.pth'
WETLAND_BOUNDARY_SHAPEFILE = r'Z:\ramsar_sweden\ramsar_polygons_by_name\\' + STUDY_AREA + '.shp'
PREDICTION_SHAPEFILES_DIR = f'prediction_shps/{STUDY_AREA}/'
PREDICTION_CSV_FILE = f'prediction_csvs/{STUDY_AREA}_water_estimates.csv'

### Get the GPU device
def get_device():
    """
    Get the appropriate device for computation (GPU/CPU).

    Returns:
        torch.device: The device to be used for computation.
    """
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Device: {}".format(device))

    if str(device) == "cuda:0":
        print("GPU: {}".format(torch.cuda.get_device_name(0)))

    return device

### Load the model
def load_model(model_file, device):
    """
    Load the pre-trained model.

    Args:
        model_file (str): Path to the model file.
        device (torch.device): The device to load the model onto.

    Returns:
        Unet: The loaded U-Net model.
    """
    model = Unet(in_channels=1, out_channels=1, init_dim=64, num_blocks=5)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()

    print(f'Model file {model_file} successfully loaded.')

    return model

### Prepare to read images
def read_image(tiff_file):
    """
    Read and normalize a TIFF image.

    Args:
        tiff_file (str): Path to the TIFF file.

    Returns:
        np.ndarray: Normalized image.
    """
    with rasterio.open(tiff_file) as src:
        image = src.read(1)  # Read the first band
        image = image.astype(np.float32)  # Ensure the image is in float32 for processing
        # Normalize image if needed
        image = (image - np.nanmin(image)) / (np.nanmax(image) - np.nanmin(image)) * 255
        return image

def generate_raster(image, src_tif, dest_file, step_size):
    """
    Generate a raster image with a specific step size.

    Args:
        image (np.ndarray): The image data to write.
        src_tif (str): Path to the source TIFF file.
        dest_file (str): Path to the destination file.
        step_size (int): Size of the patches.
    """
    with rasterio.open(src_tif) as src:
        width = src.width - src.width % step_size
        height = src.height - src.height % step_size
        window = Window(0, 0, width, height)
        transform = src.window_transform(window)

        out_meta = src.meta
        out_meta.update({
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "transform": transform
        })

        with rasterio.open(dest_file, "w", **out_meta) as dest:
            dest.write(image.astype(rasterio.uint8), 1)

def predict_water_mask(sar_image, model, device, threshold=0.5):
    """
    Predict the water mask from a SAR image using the model.

    Args:
        sar_image (np.ndarray): The input SAR image.
        model (Unet): The U-Net model for prediction.
        device (torch.device): The device to run the model on.
        threshold (float): The threshold to classify pixels as water or not.

    Returns:
        np.ndarray: The predicted water mask.
    """
    height, width = sar_image.shape
    height_adj = height - height % PATCH_SIZE
    width_adj = width - width % PATCH_SIZE
    pred_mask = np.zeros((height_adj, width_adj))

    for h in range(0, height_adj, PATCH_SIZE):
        for w in range(0, width_adj, PATCH_SIZE):
            sar_image_crop = sar_image[h:h + PATCH_SIZE, w:w + PATCH_SIZE]
            sar_image_crop = sar_image_crop[None, None, :, :]
            sar_image_crop = sar_image_crop / 255.0  # Normalize

            sar_image_crop = torch.from_numpy(sar_image_crop.astype(np.float32)).to(device)

            with torch.no_grad():
                pred = model(sar_image_crop).cpu().detach().numpy()

            pred = pred.squeeze()

            pred = np.where(pred < threshold, 0, 1)

            pred_mask[h:h + PATCH_SIZE, w:w + PATCH_SIZE] = pred
    return pred_mask

def visualize_predicted_image(image, model, device, file_name, model_name):
    """
    Visualize and save the predicted image and mask.

    Args:
        image (np.ndarray): The input SAR image.
        model (Unet): The U-Net model for prediction.
        device (torch.device): The device to run the model on.
        file_name (str): The name of the input file.
        model_name (str): The name of the model used.

    Returns:
        dict: Results including date, satellite, and file name.
    """
    width = image.shape[0] - image.shape[0] % PATCH_SIZE
    height = image.shape[1] - image.shape[1] % PATCH_SIZE

    pred_mask = predict_water_mask(image, model, device)

    unique, counts = np.unique(pred_mask, return_counts=True)
    results = dict(zip(unique, counts))
    image_date = file_name[17:25]
    satellite = file_name[0:3]
    results['Date'] = image_date
    results['Satellite'] = satellite
    results['File_name'] = file_name

    if not os.path.isdir(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Plotting SAR
    plt.imshow(image[:width, :height], cmap='gray')

    # Plotting prediction
    plt.imshow(pred_mask)
    img = Image.fromarray(np.uint8((pred_mask) * 255), 'L')

    # Find a TIFF file to generate a GeoTIFF for the prediction mask
    tif_files = [file for file in os.listdir(BULK_EXPORT_DIR) if file.endswith('.tif')]
    if tif_files:
        tif_file = os.path.join(BULK_EXPORT_DIR, tif_files[0])
    else:
        raise FileNotFoundError("No TIF files found in the directory.")

    # Generate a GeoTIFF for the prediction mask
    geotiff_file = os.path.join(RESULTS_DIR, f"{image_date}_{file_name}_pred.tif")
    generate_raster(pred_mask, tif_file, geotiff_file, PATCH_SIZE)

    return results

def get_prediction_image(tiff_file, model, device, model_name):
    """
    Process a single TIFF file and generate predictions.

    Args:
        tiff_file (str): Path to the TIFF file.
        model (Unet): The U-Net model for prediction.
        device (torch.device): The device to run the model on.
        model_name (str): The name of the model used.

    Returns:
        dict: Results including date, satellite, and file name.
    """
    image = read_image(tiff_file)
    if image is None:
        return None
    file_name = os.path.basename(tiff_file)
    visualize_predicted_image(image, model, device, file_name, model_name)

def get_pred_area():
    """
    Calculate the area of predicted water bodies and save the results to a CSV file.
    """

    utm_crs = "EPSG:32633"
    areas = []

    if not os.path.exists(PREDICTION_SHAPEFILES_DIR):
        os.makedirs(PREDICTION_SHAPEFILES_DIR)

    # Loop through all the raster files in the images_dir
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith("pred.tif"):
            raster_path = os.path.join(RESULTS_DIR, filename)
            image_date = pd.to_datetime(filename[0:8], format='%Y%m%d')

            # Use a context manager to open the raster file
            with rasterio.open(raster_path) as src:
                # Read the first band
                binary_array = src.read(1)

                # Make sure you read the transform inside the context manager
                transform = src.transform

                # Get shapes (polygons)
                shapes = list(rasterio.features.shapes(binary_array, transform=transform))

                # Extract polygons with values 1 (or adjust as needed)
                polygons = [shape(geom) for geom, value in shapes if value == 1]

                if polygons:
                    # Dissolve the polygons into a single polygon and buffer it by 0 (no buffer)
                    dissolved_polygon = MultiPolygon(polygons).buffer(0)

                    # Create a GeoDataFrame with the dissolved polygon
                    dissolved_gdf = gpd.GeoDataFrame(geometry=[dissolved_polygon])

                    # Set the CRS of the GeoDataFrame to WGS84 (EPSG:4326)
                    dissolved_gdf.crs = "EPSG:4326"

                    # Reproject the GeoDataFrame to UTM (or your desired projected CRS)
                    dissolved_gdf = dissolved_gdf.to_crs(utm_crs)

                    # Clip by wetland boundary using gpd
                    wetland_boundary = gpd.read_file(WETLAND_BOUNDARY_SHAPEFILE)
                    wetland_boundary = wetland_boundary.to_crs(utm_crs)
                    dissolved_gdf = gpd.clip(dissolved_gdf, wetland_boundary)

                    # Save the shapefile for testing purposes
                    dissolved_gdf.to_file(os.path.join(PREDICTION_SHAPEFILES_DIR, f"{filename[:-13]}_pred.shp"))

                    # Skip if dataframe is empty
                    if dissolved_gdf.empty:
                        continue

                    # Calculate the area in square meters (m²) in the projected CRS
                    area_utm = dissolved_gdf.geometry.area.iloc[0]

                    # Optionally, you can print the area for each raster
                    print(f"Area of {filename} (UTM): {area_utm} m²")

                    # Store the area information in the list
                    areas.append({'Name': STUDY_AREA, 'Date': image_date, 'Area (metres squared)': area_utm})

    # Create a CSV file with the area information for all images
    with open(PREDICTION_CSV_FILE, 'w', newline='') as csvfile:
        fieldnames = ['Name', 'Date', 'Area (metres squared)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(areas)

def full_cycle(model_name):
    """
    Execute the full prediction cycle including loading the model, processing TIFFs, and calculating areas.

    Args:
    model_name (str): The name of the model used.
    """

    if not os.path.exists(BULK_EXPORT_DIR):
        raise FileNotFoundError(f'The folder containing the TIFF files does not exist: {BULK_EXPORT_DIR}')

    filenames = [f for f in os.listdir(BULK_EXPORT_DIR) if f.endswith('.tif')]

    device = get_device()
    model_file = PRETRAINED_MODEL_DIR
    model = load_model(model_file, device)

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Initialize the progress bar with the number of files
    for tiff_file in tqdm(filenames, desc="Processing TIFFs"):
        get_prediction_image(os.path.join(BULK_EXPORT_DIR, tiff_file), model, device, model_name)

    print("Completed all predictions.")

    # Generate the area estimates
    print("Generating area estimates...")
    get_pred_area()

def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    full_cycle(MODEL_NAME)

if __name__ == "__main__":
    main()
