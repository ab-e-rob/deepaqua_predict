import os
import torch
import numpy as np
import rasterio as rio
from models.unet import Unet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import geopandas as gpd
from rasterio.features import rasterize
import wandb
from rasterio.windows import Window

# This code's purpose is to test any new models that are created in the training loop
# using manually annotated data.

STUDY_AREA = 'helge'
STUDY_YEAR = '2023'  # Example study year
PATCH_SIZE = 64
MODEL_PATH = 'pretrained_models/spring-jazz-85_epoch_60.pth'
RESULTS_DIR = 'validation_data/'
VAL_IMAGE_DIR = 'validation_data/sar_imagery_val/'
VAL_LABEL_DIR = 'validation_data/manual_labels_val/'

def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    return device

def load_model(model_path, device):
    model = Unet(in_channels=3, out_channels=1, init_dim=64, num_blocks=5)  # Assuming 3 bands input (VV, VH, VV-VH)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f'Model file {model_path} successfully loaded.')
    return model

def find_sar_image(directory):
    """Locate the SAR image file that matches the study area and date."""
    for filename in os.listdir(directory):
        if filename.endswith('.tif') and STUDY_AREA in filename and STUDY_YEAR in filename:
            return os.path.join(directory, filename)
    raise FileNotFoundError(f'No matching SAR image found in {directory} for area "{STUDY_AREA}" and date "{STUDY_YEAR}".')

def read_image(tiff_file, ignore_nan=False):
    with rio.open(tiff_file) as src:
        image = np.zeros((3, src.height, src.width), dtype=np.float32)
        image[0, :, :] = src.read(1).astype(np.float32)  # VV
        image[1, :, :] = src.read(2).astype(np.float32)  # VH
        image[2, :, :] = image[0, :, :] - image[1, :, :]  # VV - VH (Difference)

        if ignore_nan and np.isnan(image).any():
            return None

        for i in range(3):
            min_value = np.nanpercentile(image[i, :, :], 1)
            max_value = np.nanpercentile(image[i, :, :], 99)
            image[i, :, :] = np.clip(image[i, :, :], min_value, max_value)
            image[i, :, :] = (image[i, :, :] - min_value) / (max_value - min_value)
            image[i, :, :][np.isnan(image[i, :, :])] = 0

        return image

def generate_raster(image, dest_file, src_tif):
    with rio.open(src_tif) as src:
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": image.shape[0],
            "width": image.shape[1],
            "count": 1,
            "dtype": 'uint8'
        })

        with rio.open(dest_file, "w", **out_meta) as dest:
            dest.write(image.astype(rio.uint8), 1)

def predict_water_mask(sar_image, model, device, threshold=0.5):
    channels, height, width = sar_image.shape
    height_adj = height - height % PATCH_SIZE
    width_adj = width - width % PATCH_SIZE

    sar_image = sar_image[:, :height_adj, :width_adj]
    pred_mask = np.zeros((height_adj, width_adj))

    for h in range(0, height_adj, PATCH_SIZE):
        for w in range(0, width_adj, PATCH_SIZE):
            sar_image_crop = sar_image[:, h:h + PATCH_SIZE, w:w + PATCH_SIZE]
            binary_image = np.where(sar_image_crop.sum(0) > 0, 1, 0)
            sar_image_crop = torch.from_numpy(sar_image_crop.astype(np.float32)).to(device)
            pred = model(sar_image_crop[None, ...]).cpu().detach().numpy().squeeze()

            pred = (pred * binary_image)
            pred_mask[h:h + PATCH_SIZE, w:w + PATCH_SIZE] = np.where(pred < threshold, 0, 1)

    return pred_mask

def visualize_predicted_image(image, model, device, file_name, model_name):
    channels, height, width = image.shape
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

    geotiff_file = os.path.join(RESULTS_DIR, f"{image_date}_{file_name}_pred_v.tif")
    generate_raster(pred_mask, geotiff_file, file_name)  # Corrected to use the file name

    return results

def get_prediction_image(tiff_file, model, device, model_name):
    image = read_image(tiff_file)
    if image is None:
        return None
    file_name = os.path.basename(tiff_file)
    visualize_predicted_image(image, model, device, file_name, model_name)

def shp_to_raster(shapefile, reference_raster):
    with rio.open(reference_raster) as src:
        transform = src.transform
        out_shape = (src.height, src.width)

    # Read the shapefile
    gdf = gpd.read_file(shapefile)

    # Check and reproject if CRS does not match
    if gdf.crs != src.crs:
        gdf = gdf.to_crs(src.crs)

    # Rasterize the water geometries
    rasterized = rasterize(
        [(geom, 1) for geom in gdf.geometry],  # Assign 1 to water geometries
        out_shape=out_shape,
        transform=transform,
        fill=0,  # Fill background with 0
        dtype=np.uint8
    )

    return rasterized

def calculate_metrics(pred_mask, true_mask):
    accuracy = accuracy_score(true_mask.flatten(), pred_mask.flatten())
    precision = precision_score(true_mask.flatten(), pred_mask.flatten(), zero_division=0)
    recall = recall_score(true_mask.flatten(), pred_mask.flatten(), zero_division=0)
    f1 = f1_score(true_mask.flatten(), pred_mask.flatten(), zero_division=0)
    iou = jaccard_score(true_mask.flatten(), pred_mask.flatten(), zero_division=0)
    return accuracy, precision, recall, f1, iou

def save_prediction_as_tiff(pred_mask, reference_tiff, output_tiff):
    with rio.open(reference_tiff) as src:
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": pred_mask.shape[0],
            "width": pred_mask.shape[1],
            "count": 1,
            "dtype": 'uint8'
        })

        with rio.open(output_tiff, "w", **out_meta) as dest:
            dest.write(pred_mask.astype(rio.uint8), 1)

def test(model, device):
    if not os.path.exists(VAL_IMAGE_DIR):
        raise FileNotFoundError(f'The folder containing the TIFF files does not exist: {VAL_IMAGE_DIR}')

    # Find the correct SAR image file
    file_path = find_sar_image(VAL_IMAGE_DIR)
    image = read_image(file_path)

    metrics = []

    # Optionally compare with manual annotations
    manual_annotation_path = os.path.join(VAL_LABEL_DIR, f"{os.path.basename(file_path)[:-4]}.shp")
    if os.path.exists(manual_annotation_path):
        true_mask = shp_to_raster(manual_annotation_path, file_path)

        # Generate the predicted mask
        pred_mask = predict_water_mask(image, model, device)

        # Crop the ground truth mask to the shape of the predicted mask
        true_mask_cropped = true_mask[:pred_mask.shape[0], :pred_mask.shape[1]]

        # Save the prediction mask as a TIFF file
        output_tiff_path = os.path.join(RESULTS_DIR, f"{os.path.basename(file_path)[:-4]}_prediction.tif")
        save_prediction_as_tiff(pred_mask, file_path, output_tiff_path)

        # Calculate metrics using the cropped ground truth mask
        acc, prec, rec, f1, iou = calculate_metrics(pred_mask, true_mask_cropped)
        metrics.append({
            'file': os.path.basename(file_path),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'iou': iou
        })

    # Print metrics for the processed file
    for metric in metrics:
        print(f"Metrics for {metric['file']}:")
        print(f"Accuracy: {metric['accuracy']:.4f}, Precision: {metric['precision']:.4f}, "
              f"Recall: {metric['recall']:.4f}, F1 Score: {metric['f1_score']:.4f}, IOU: {metric['iou']:.4f}")

    print("Testing complete!")


if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="water_mask_validation", entity="abigail-robinson")

    device = get_device()
    model = load_model(MODEL_PATH, device)
    test(model, device)
