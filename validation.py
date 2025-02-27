import os
import torch
import numpy as np
import rasterio as rio
from models.unet import Unet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import geopandas as gpd
from rasterio.features import rasterize
import wandb
import pandas as pd
import matplotlib.pyplot as plt

STUDY_AREA = 'osten'
STUDY_YEAR = '2024'
PATCH_SIZE = 64
MODEL_PATH = 'pretrained_models/model_2024.pth'
RESULTS_DIR = 'validation_data/'
VAL_IMAGE_DIR = 'validation_data/sar_imagery_val/'
VAL_LABEL_DIR = 'validation_data/manual_labels_val/'

def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    return device

def load_model(model_path, device):
    model = Unet(in_channels=1, out_channels=1, init_dim=64, num_blocks=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f'Model file {model_path} successfully loaded.')
    return model

def find_sar_image(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.tif') and STUDY_AREA in filename and STUDY_YEAR in filename:
            return os.path.join(directory, filename)
    raise FileNotFoundError(f'No matching SAR image found in {directory} for area "{STUDY_AREA}" and date "{STUDY_YEAR}".')

def read_image(tiff_file, ignore_nan=False):
    with rio.open(tiff_file) as src:
        image = np.zeros((1, src.height, src.width), dtype=np.float32)
        image[0, :, :] = src.read(2).astype(np.float32)  # VH only

        if ignore_nan and np.isnan(image).any():
            return None

        for i in range(1):
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

    pred_mask = np.zeros((height, width))

    for h in range(0, height, PATCH_SIZE):
        for w in range(0, width, PATCH_SIZE):
            h_end = min(h + PATCH_SIZE, height)
            w_end = min(w + PATCH_SIZE, width)
            sar_image_crop = sar_image[:, h:h_end, w:w_end]

            if sar_image_crop.shape[1:] != (PATCH_SIZE, PATCH_SIZE):
                padding = ((0, 0),
                           (0, PATCH_SIZE - sar_image_crop.shape[1]),
                           (0, PATCH_SIZE - sar_image_crop.shape[2]))
                sar_image_crop = np.pad(sar_image_crop, padding, mode='constant', constant_values=0)

            sar_image_crop = torch.from_numpy(sar_image_crop.astype(np.float32)).to(device)
            sar_image_crop = sar_image_crop.unsqueeze(0)
            pred = model(sar_image_crop).cpu().detach().numpy().squeeze()
            pred = np.where(pred < threshold, 0, 1)

            pred_height = h_end - h
            pred_width = w_end - w
            pred_mask[h:h_end, w:w_end] = pred[:pred_height, :pred_width]

    return pred_mask

def visualize_predicted_image(image, model, device, file_name, src_tif):
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

    geotiff_file = os.path.join(RESULTS_DIR, f"{file_name}_pred.tif")

    print(f"Saving predicted mask to: {geotiff_file}")  # Debugging print statement
    generate_raster(pred_mask, geotiff_file, src_tif)

    return results
def get_prediction_image(tiff_file, model, device):
    image = read_image(tiff_file)
    if image is None:
        return None
    file_name = os.path.basename(tiff_file)
    visualize_predicted_image(image, model, device, file_name, tiff_file)

def shp_to_raster(shapefile, reference_raster):
    with rio.open(reference_raster) as src:
        transform = src.transform
        out_shape = (src.height, src.width)

    gdf = gpd.read_file(shapefile)

    if gdf.crs != src.crs:
        gdf = gdf.to_crs(src.crs)

    rasterized = rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=out_shape,
        transform=transform,
        fill=0,
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

def plot_validation_patches(input_image, true_mask, pred_mask, num_patches=10):
    plt.figure(figsize=(15, 5 * num_patches))
    height, width = input_image.shape[1:3]

    for i in range(num_patches):
        h = np.random.randint(0, height - PATCH_SIZE)
        w = np.random.randint(0, width - PATCH_SIZE)

        input_patch = input_image[:, h:h + PATCH_SIZE, w:w + PATCH_SIZE]
        true_patch = true_mask[h:h + PATCH_SIZE, w:w + PATCH_SIZE]
        pred_patch = pred_mask[h:h + PATCH_SIZE, w:w + PATCH_SIZE]

        plt.subplot(num_patches, 3, 3 * i + 1)
        plt.imshow(input_patch[0], cmap='gray')
        plt.title('Input Image Patch')

        plt.subplot(num_patches, 3, 3 * i + 2)
        plt.imshow(true_patch, cmap='gray')
        plt.title('Ground Truth Patch')

        plt.subplot(num_patches, 3, 3 * i + 3)
        plt.imshow(pred_patch, cmap='gray')
        plt.title('Prediction Patch (Validation)')

    plt.tight_layout()
    plt.show()

def test(model, device):
    if not os.path.exists(VAL_IMAGE_DIR):
        raise FileNotFoundError(f'The folder containing the TIFF files does not exist: {VAL_IMAGE_DIR}')

    file_path = find_sar_image(VAL_IMAGE_DIR)

    image = read_image(file_path)

    get_prediction_image(file_path, model, device)

    metrics = []

    manual_annotation_path = os.path.join(VAL_LABEL_DIR, f"{os.path.basename(file_path)[:-4]}.shp")
    if os.path.exists(manual_annotation_path):
        true_mask = shp_to_raster(manual_annotation_path, file_path)

        pred_mask = predict_water_mask(image, model, device)

        true_mask_cropped = true_mask[:pred_mask.shape[0], :pred_mask.shape[1]]

        acc, prec, rec, f1, iou = calculate_metrics(pred_mask, true_mask_cropped)
        metrics.append([os.path.basename(file_path), acc, prec, rec, f1, iou])

        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}, IoU: {iou:.4f}")

        plot_validation_patches(image, true_mask_cropped, pred_mask)

    return metrics

def main():
    device = get_device()
    model = load_model(MODEL_PATH, device)
    metrics = test(model, device)

    df = pd.DataFrame(metrics, columns=['Image', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'IoU'])
    df.to_csv(os.path.join(RESULTS_DIR, 'validation_metrics.csv'), index=False)

if __name__ == '__main__':
    main()
