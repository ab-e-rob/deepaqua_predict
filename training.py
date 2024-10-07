import os
import wandb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
from tqdm.notebook import tqdm
from models.unet import Unet

### SET PARAMETERS
config = {
    'EPOCHS': 20,
    'PATCH_SIZE': 64,
    'BATCH_SIZE': 32,
    'NUM_WORKERS': 4,
    'RANDOM_SEED': 42,
    'LEARNING_RATE': 0.0001,
    'MODELS_DIR': 'models',
    'NEW_MODELS_DIR': 'new_models',
    'LOSS_FUNCTION': 'dice',
    'CNN_TYPE': 'unet',
    'SAR_DIR': 'training_data/sar_tiles/',
    'NDWI_MASK_DIR': 'training_data/ndwi_tiles/',
    'TILES_FILE': 'tiles.csv'  # Changed to .csv for consistency
}

class DiceLoss(nn.Module):
    def __init__(self, lambda_=1.):
        super(DiceLoss, self).__init__()
        self.lambda_ = lambda_

    def forward(self, y_pred, y_true):
        y_pred = y_pred[:, 0].view(-1)
        y_true = y_true[:, 0].view(-1)
        intersection = (y_pred * y_true).sum()
        dice_loss = (2. * intersection + self.lambda_) / (y_pred.sum() + y_true.sum() + self.lambda_)
        return 1. - dice_loss

def create_loss_function(loss_function_name):
    if loss_function_name == 'dice':
        return DiceLoss()
    raise ValueError(f'Unrecognized loss function: {loss_function_name}')

def load_model(model_file, device):
    model = Unet(in_channels=1, out_channels=1, init_dim=64, num_blocks=5)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    print(f'Model file {model_file} successfully loaded.')
    return model

def get_device():
    """Get the appropriate device for computation (GPU/CPU)."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Simplified for clarity
    print("Device: {}".format(device))
    if device.type == "cuda":
        print("GPU: {}".format(torch.cuda.get_device_name(0)))
    return device

def create_tiles_file(sar_dir, ndwi_dir):
    sar_files = [f for f in os.listdir(sar_dir) if f.endswith('.tif')]
    ndwi_files = [f for f in os.listdir(ndwi_dir) if f.endswith('.tif')]

    # Create a set of base names for comparison
    sar_base_names = {f.replace('-sar_image.tif', '') for f in sar_files}
    ndwi_base_names = {f.replace('-ndwi_mask.tif', '') for f in ndwi_files}

    # Find common base names
    common_base_names = sar_base_names.intersection(ndwi_base_names)

    if not common_base_names:
        raise ValueError("No common files found between SAR and NDWI directories.")

    # Create common files list based on the base names
    common_files = [
        f for f in sar_files
        if f.replace('-sar_image.tif', '') in common_base_names
    ]

    # Extract numeric indexes from common files
    common_indexes = [
        int(f.split('-')[1])  # Extract the number from "tile-10001-sar_image.tif"
        for f in common_files
    ]

    tiles_data_frame = pd.DataFrame({'index': common_indexes, 'id': common_files})
    tiles_data_frame.set_index('index', inplace=True)
    tiles_data_frame['split'] = 'train'

    # reduce the number of tiles for testing by 90%
    tiles_data_frame = tiles_data_frame.sample(frac=1)

    # Split into train and test sets
    num_rows = len(tiles_data_frame)
    test_rows = int(num_rows * 0.2)
    tiles_data_frame.loc[tiles_data_frame.tail(test_rows).index, 'split'] = 'test'

    # Save to CSV
    tiles_file = os.getenv("TILES_FILE", "tiles.csv")  # Default to "tiles.csv" if not set
    tiles_data_frame.to_csv(tiles_file, columns=['id', 'split'], index_label='index')
    print('Total tiles:', num_rows)

def plant_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

class CFDDataset(Dataset):
    def __init__(self, dataset, images_dir, masks_dir):
        self.dataset = dataset
        self.images_dir = images_dir
        self.masks_dir = masks_dir

    def __getitem__(self, index):
        index_ = self.dataset.iloc[index]['id']
        # Adjust the image_path and mask_path based on the correct naming convention
        image_path = os.path.join(self.images_dir, f"{index_}")
        mask_path = os.path.join(self.masks_dir, f"{index_.replace('-sar_image.tif', '-ndwi_mask.tif')}")

        # Ensure the correct extension is included
        if not image_path.endswith('.tif'):
            image_path += '-sar_image.tif'  # Ensure the correct suffix for SAR images
        if not mask_path.endswith('.tif'):
            mask_path += '-ndwi_mask.tif'  # Ensure the correct suffix for NDWI masks

        # Load the image and mask
        image = rio.open(image_path).read()
        mask = rio.open(mask_path).read()
        return torch.from_numpy(image.astype(np.float32)), torch.from_numpy(mask.astype(np.float32))

    def __len__(self):
        return len(self.dataset)

def get_dataloaders(data, batch_size, num_workers, images_dir, masks_dir):
    datasets = {
        'train': CFDDataset(data[data.split == 'train'], images_dir, masks_dir),
        'test': CFDDataset(data[data.split == 'test'], images_dir, masks_dir)
    }
    return {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'test': DataLoader(datasets['test'], batch_size=batch_size, drop_last=False, num_workers=num_workers)
    }

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    losses, ious = [], []
    for input, target in tqdm(dataloader, total=len(dataloader)):
        input, target = input.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(input)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        iou = intersection_over_union(output, target)

        losses.append(loss.item())
        ious.append(iou.cpu().item())

    return np.mean(losses), np.mean(ious)

def test(model, dataloader, criterion, device):
    model.eval()
    losses, ious = [], []
    inf_tile_ids = []  # To keep track of tile IDs with inf IoU
    with torch.no_grad():
        for input, target in tqdm(dataloader, total=len(dataloader)):
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = criterion(output, target)

            # Skip batches with NaN values in loss
            if torch.isnan(loss).any():
                print("Skipping batch due to NaN loss.")
                continue

            iou = intersection_over_union(output, target)

            losses.append(loss.item())
            ious.append(iou.cpu().item())

    return np.mean(losses), np.mean(ious), inf_tile_ids

def save_model(model, model_dir, model_name, epoch):
    os.makedirs(model_dir, exist_ok=True)
    model_file = f"{model_name}_epoch_{epoch}.pth"  # Use the WandB model name and epoch number
    model_path = os.path.join(model_dir, model_file)
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

def intersection_over_union(y_pred, y_true):
    smooth = 1e-6
    y_pred = (y_pred > 0.5).float()
    y_true = (y_true > 0.5).float()

    intersection = (y_pred * y_true).sum() + smooth
    union = (y_pred + y_true).sum() - intersection + smooth

    return intersection / union

def full_cycle():
    for key in ['EPOCHS', 'PATCH_SIZE', 'BATCH_SIZE', 'NUM_WORKERS', 'RANDOM_SEED']:
        config[key] = int(config[key])
    for key in ['LEARNING_RATE']:
        config[key] = float(config[key])

    wandb.init(project="deep-wetlands", entity="abigail-robinson", config=config)

    # Retrieve model name from WandB config
    model_name = wandb.run.name  # This will be the WandB run name

    for key, value in config.items():
        os.environ[key] = str(value)

    plant_random_seed(config['RANDOM_SEED'])
    create_tiles_file(config['SAR_DIR'], config['NDWI_MASK_DIR'])
    tiles_data = pd.read_csv(config['TILES_FILE'])

    device = get_device()  # Use the get_device function to set the device
    dataloaders = get_dataloaders(tiles_data, config['BATCH_SIZE'], config['NUM_WORKERS'], config['SAR_DIR'], config['NDWI_MASK_DIR'])
    model = Unet(in_channels=1, out_channels=1, init_dim=64, num_blocks=5).to(device)
    criterion = create_loss_function(config['LOSS_FUNCTION'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])

    for epoch in tqdm(range(1, config['EPOCHS'] + 1)):
        print(f'Epoch {epoch}/{config["EPOCHS"]}')
        train_loss, train_iou = train(model, dataloaders['train'], criterion, optimizer, device)
        test_loss, test_iou, inf_tile_ids = test(model, dataloaders['test'], criterion, device)

        wandb.log({
            'train/loss': train_loss,
            'train/iou': train_iou,
            'test/loss': test_loss,
            'test/iou': test_iou
        })

        print(f'train_loss: {train_loss:.4f}, train_iou: {train_iou:.4f}, test_loss: {test_loss:.4f}, test_iou: {test_iou:.4f}')

        save_model(model, config['NEW_MODELS_DIR'], model_name, epoch)

    wandb.finish()

if __name__ == "__main__":
    full_cycle()
