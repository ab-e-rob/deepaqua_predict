import os
import torch
import numpy as np
import rasterio as rio
from models.unet import Unet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import argparse

### This code's prupose is to test any new models that are created in the training loop
### using manually annotated data.

# Set parameters
PATCH_SIZE = 64
MODEL_DIR = 'new_models/big-2020.pth'
BULK_EXPORT_DIR = 'sar_imagery/Helge_sar_export'
RESULTS_DIR = 'prediction_tiffs/Helge/'
MANUAL_ANNOTATIONS_DIR = 'manual_annotations/'  # Directory containing manually annotated data


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    return device


def load_model(model_file, device):
    model = Unet(in_channels=2, out_channels=1, init_dim=64, num_blocks=5)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    print(f'Model file {model_file} successfully loaded.')
    return model


def read_image(tiff_file):
    with rio.open(tiff_file) as src:
        return src.read(1)  # Assuming you want to read the first band


def predict_water_mask(sar_image, model, device):
    height, width = sar_image.shape
    height_adj = height - height % PATCH_SIZE
    width_adj = width - width % PATCH_SIZE
    pred_mask = np.zeros((height_adj, width_adj))

    for h in range(0, height_adj, PATCH_SIZE):
        for w in range(0, width_adj, PATCH_SIZE):
            sar_image_crop = sar_image[h:h + PATCH_SIZE, w:w + PATCH_SIZE]
            sar_image_crop = sar_image_crop[None, None, :, :]  # Add batch and channel dimensions
            sar_image_crop = torch.from_numpy(sar_image_crop.astype(np.float32)).to(device)

            with torch.no_grad():
                pred = model(sar_image_crop).cpu().numpy()
            pred_mask[h:h + PATCH_SIZE, w:w + PATCH_SIZE] = (pred.squeeze() > 0.5).astype(np.uint8)

    return pred_mask


def load_manual_annotations(file_path):
    with rio.open(file_path) as src:
        return src.read(1)  # Load the manually annotated data


def calculate_metrics(pred_mask, true_mask):
    accuracy = accuracy_score(true_mask.flatten(), pred_mask.flatten())
    precision = precision_score(true_mask.flatten(), pred_mask.flatten(), zero_division=0)
    recall = recall_score(true_mask.flatten(), pred_mask.flatten(), zero_division=0)
    f1 = f1_score(true_mask.flatten(), pred_mask.flatten(), zero_division=0)
    return accuracy, precision, recall, f1


def test(model, device):
    if not os.path.exists(BULK_EXPORT_DIR):
        raise FileNotFoundError(f'The folder containing the TIFF files does not exist: {BULK_EXPORT_DIR}')

    filenames = [f for f in os.listdir(BULK_EXPORT_DIR) if f.endswith('.tif')]
    metrics = []

    for tiff_file in tqdm(filenames, desc="Testing"):
        file_path = os.path.join(BULK_EXPORT_DIR, tiff_file)
        image = read_image(file_path)
        pred_mask = predict_water_mask(image, model, device)

        # Optionally compare with manual annotations
        manual_annotation_path = os.path.join(MANUAL_ANNOTATIONS_DIR, f"{tiff_file[:-4]}_manual.tif")
        if os.path.exists(manual_annotation_path):
            true_mask = load_manual_annotations(manual_annotation_path)
            acc, prec, rec, f1 = calculate_metrics(pred_mask, true_mask)
            metrics.append({'file': tiff_file, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1})

    # Print metrics for each file
    for metric in metrics:
        print(f"Metrics for {metric['file']}:")
        print(f"Accuracy: {metric['accuracy']:.4f}, Precision: {metric['precision']:.4f}, "
              f"Recall: {metric['recall']:.4f}, F1 Score: {metric['f1_score']:.4f}")

    print("Testing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the water mask prediction model.")
    parser.add_argument('--model', type=str, default=MODEL_DIR, help='Path to the pretrained model')
    args = parser.parse_args()

    device = get_device()
    model = load_model(args.model, device)
    test(model, device)
