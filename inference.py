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
import rasterio as rio

### Set parameters
STUDY_AREA = 'Kulbacksliden'
#MODEL_NAME = 'big-2020'
PATCH_SIZE = 64
BULK_EXPORT_DIR = f'sar_imagery/{STUDY_AREA}_sar_export'
RESULTS_DIR = f'prediction_tiffs/{STUDY_AREA}/'
#PRETRAINED_MODEL_DIR = f'pretrained_models/{MODEL_NAME}.pth'
WETLAND_BOUNDARY_SHAPEFILE = r'study_areas/' + STUDY_AREA + '.shp'
PREDICTION_SHAPEFILES_DIR = f'prediction_shps/{STUDY_AREA}/'
PREDICTION_CSV_FILE = f'prediction_csvs/{STUDY_AREA}_water_estimates.csv'

### Get the GPU device
def get_device():
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Device: {}".format(device))

    if str(device) == "cuda:0":
        print("GPU: {}".format(torch.cuda.get_device_name(0)))

    return device

### Load the model
def load_model(model_file, device):
    model = Unet(in_channels=1, out_channels=1, init_dim=64, num_blocks=5)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()

    print(f'Model file {model_file} successfully loaded.')

    return model

### Read and normalize image
def read_image(tiff_file, ignore_nan=False):
    with rio.open(tiff_file) as src:
        #image = np.zeros((2, src.height, src.width), dtype=np.float32)
        image = np.zeros((1, src.height, src.width), dtype=np.float32)  # Only one band
        #image[0, :, :] = src.read(1).astype(np.float32)  # VV
        #image[1, :, :] = src.read(2).astype(np.float32)  # VH
        image[0, :, :] = src.read(2).astype(np.float32)  # VH

        if ignore_nan and np.isnan(image).any():
            return None

        for i in range(1):
            min_value = np.nanpercentile(image[i, :, :], 1)
            max_value = np.nanpercentile(image[i, :, :], 99)
            image[i, :, :] = np.clip(image[i, :, :], min_value, max_value)
            image[i, :, :] = (image[i, :, :] - min_value) / (max_value - min_value)
            image[i, :, :][np.isnan(image[i, :, :])] = 0

        return image

### Generate raster from prediction mask
def generate_raster(image, src_tif, dest_file, step_size):
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

### Predict water mask
def predict_water_mask(sar_image, model, device, threshold=0.5):
    channels, height, width = sar_image.shape  # Assuming sar_image has shape (channels, height, width)

    # Adjust width and height to be divisible by PATCH_SIZE
    width = width - width % PATCH_SIZE
    height = height - height % PATCH_SIZE

    pred_mask = np.zeros(tuple((height, width)))

    for h in range(0, height, PATCH_SIZE):
        for w in range(0, width, PATCH_SIZE):
            # Correct slicing to get a patch of size [64, 64]
            sar_image_crop = sar_image[:, h:h + PATCH_SIZE, w:w + PATCH_SIZE]

            # Ensure the cropped image patch has the correct dimensions
            if sar_image_crop.shape[1:] != (PATCH_SIZE, PATCH_SIZE):
                # Pad the image if the patch size is smaller than expected
                padding = ((0, 0),
                           (0, PATCH_SIZE - sar_image_crop.shape[1]),
                           (0, PATCH_SIZE - sar_image_crop.shape[2]))
                sar_image_crop = np.pad(sar_image_crop, padding, mode='constant', constant_values=0)

            # Convert to tensor and move to device
            sar_image_crop = torch.from_numpy(sar_image_crop.astype(np.float32)).to(device)

            # Add batch and channel dimensions for model input
            sar_image_crop = sar_image_crop.unsqueeze(0)

            # Model prediction
            pred = model(sar_image_crop).cpu().detach().numpy().squeeze()

            # Apply threshold
            pred = np.where(pred < threshold, 0, 1)

            # Assign prediction to the mask
            pred_mask[h:h + PATCH_SIZE, w:w + PATCH_SIZE] = pred[:PATCH_SIZE, :PATCH_SIZE]

    return pred_mask

### Visualize and save predicted image
def visualize_predicted_image(image, model, device, file_name, model_name):
    channels, height, width = image.shape

    width = width - width % PATCH_SIZE
    height = height - height % PATCH_SIZE

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

    plt.imshow(image[0, :width, :height], cmap='gray')
    plt.imshow(pred_mask, alpha=0.5)
    img = Image.fromarray(np.uint8((pred_mask) * 255), 'L')

    tif_files = [file for file in os.listdir(BULK_EXPORT_DIR) if file.endswith('.tif')]
    if tif_files:
        tif_file = os.path.join(BULK_EXPORT_DIR, tif_files[0])
    else:
        raise FileNotFoundError("No TIF files found in the directory.")

    geotiff_file = os.path.join(RESULTS_DIR, f"{image_date}_{file_name}_pred.tif")
    generate_raster(pred_mask, tif_file, geotiff_file, PATCH_SIZE)

    return results

### Predict image
def get_prediction_image(tiff_file, model, device, model_name):
    image = read_image(tiff_file)
    if image is None:
        return None
    file_name = os.path.basename(tiff_file)
    visualize_predicted_image(image, model, device, file_name, model_name)

### Calculate predicted area
def get_pred_area():
    utm_crs = "EPSG:32633"
    areas = []

    if not os.path.exists(PREDICTION_SHAPEFILES_DIR):
        os.makedirs(PREDICTION_SHAPEFILES_DIR)

    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith("pred.tif"):
            raster_path = os.path.join(RESULTS_DIR, filename)
            image_date = pd.to_datetime(filename[0:8], format='%Y%m%d')

            with rasterio.open(raster_path) as src:
                binary_array = src.read(1)
                transform = src.transform
                shapes = list(rasterio.features.shapes(binary_array, transform=transform))
                polygons = [shape(geom) for geom, value in shapes if value == 1]

                if polygons:
                    dissolved_polygon = MultiPolygon(polygons).buffer(0)
                    dissolved_gdf = gpd.GeoDataFrame(geometry=[dissolved_polygon])
                    dissolved_gdf.crs = "EPSG:4326"
                    dissolved_gdf = dissolved_gdf.to_crs(utm_crs)
                    wetland_boundary = gpd.read_file(WETLAND_BOUNDARY_SHAPEFILE)
                    wetland_boundary = wetland_boundary.to_crs(utm_crs)
                    dissolved_gdf = gpd.clip(dissolved_gdf, wetland_boundary)
                    dissolved_gdf.to_file(
                        os.path.join(PREDICTION_SHAPEFILES_DIR, f"{filename[:-13]}_pred.shp"))

                    if dissolved_gdf.empty:
                        continue

                    area_utm = dissolved_gdf.geometry.area.iloc[0]
                    print(f"Area of {filename} (UTM): {area_utm} m²")
                    areas.append({'Name': STUDY_AREA, 'Date': image_date, 'Area (metres squared)': area_utm})

    with open(PREDICTION_CSV_FILE, 'w', newline='') as csvfile:
        fieldnames = ['Name', 'Date', 'Area (metres squared)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(areas)

def get_model_name_based_on_year(file_name):
    # Extract year from file name (assuming the year is part of the filename)
    image_date = file_name[17:25]  # Adjust this based on your file naming convention
    year = int(image_date[:4])

    # Set model name based on the year
    if year in [2020, 2021, 2022, 2023]:
        return 'big-2020'
    elif year in [2018, 2019]:
        return 'big-2018_best_model'
    elif year == 2024:
        return 'model_2024'
    else:
        raise ValueError(f"Unsupported image year: {year}")

### Full cycle
def full_cycle():
    if not os.path.exists(BULK_EXPORT_DIR):
        raise FileNotFoundError(f'The folder containing the TIFF files does not exist: {BULK_EXPORT_DIR}')

    filenames = [f for f in os.listdir(BULK_EXPORT_DIR) if f.endswith('.tif')]

    device = get_device()

    for tiff_file in tqdm(filenames, desc="Processing SAR images"):
        # Get model name based on the year of the image
        model_name = get_model_name_based_on_year(tiff_file)
        model_file = f'pretrained_models/{model_name}.pth'
        model = load_model(model_file, device)
        get_prediction_image(os.path.join(BULK_EXPORT_DIR, tiff_file), model, device, model_name)

    get_pred_area()

if __name__ == '__main__':
    full_cycle()
