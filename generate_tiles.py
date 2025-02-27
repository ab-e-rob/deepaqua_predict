import os
import numpy as np
import rasterio as rio
import rasterio.mask
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import glob
import matplotlib.pyplot as plt

### Parameters

NDWI_MASK_DIR = 'training_data/ndwi_mask'
CROPPED_NDWI_MASK_DIR = 'training_data/ndwi_tiles'
SAR_DIR = 'training_data/sar'
CROPPED_SAR_DIR = 'training_data/sar_tiles'
PATCH_SIZE = 64
EPSG = 32633

tile_id_counter = 0  # Global tile ID counter

def generate_tiles(image_file, size):
    """Generates size x size polygon tiles."""
    global tile_id_counter
    with rio.open(image_file) as raster:
        width, height = raster.width, raster.height
        geo_dict = {'id': [], 'geometry': []}

        for w in tqdm(range(0, width, size), total=width // size):
            for h in range(0, height, size):
                if w + size > width or h + size > height:
                    continue  # Skip tiles that would exceed image bounds

                window = rio.windows.Window(w, h, size, size)
                bbox = rio.windows.bounds(window, raster.transform)
                geo_dict['id'].append(f'tile-{tile_id_counter}')
                geo_dict['geometry'].append(box(*bbox))
                tile_id_counter += 1

        return gpd.GeoDataFrame(pd.DataFrame(geo_dict), crs=f"epsg:{EPSG}")

def export_ndwi_mask_data(tiles, tif_file):
    if not os.path.exists(CROPPED_NDWI_MASK_DIR):
        os.makedirs(CROPPED_NDWI_MASK_DIR)

    with rio.open(tif_file) as src:
        # Read and flip the SAR image
        dataset_array = src.read()

        minValue = np.nanpercentile(dataset_array, 1)
        maxValue = np.nanpercentile(dataset_array, 99)

        # plot
        plt.imshow(dataset_array[0], cmap='gray')
        plt.show()

    # Process each tile
    for _, tile in tqdm(tiles.iterrows(), total=tiles.shape[0]):

        with rio.open(tif_file) as src:

            shape = [tile['geometry']]
            name = tile['id']
            print(f"Processing SAR tile: {name}")

            try:
                # Mask the flipped image for the current tile
                out_image, out_transform = rio.mask.mask(src, shape, crop=True)
                # out_image = out_image.astype(np.float32)

                if np.isnan(out_image).any():
                    print(f"Skipping ndwi tile {name} due to NaN values.")
                    continue

                if out_image.shape[1] < PATCH_SIZE or out_image.shape[2] < PATCH_SIZE:
                    print(f"Skipping ndwi tile {name} due to small size: {out_image.shape}")
                    continue

                if out_image.shape[1] == PATCH_SIZE + 1:
                    out_image = out_image[:, :-1, :]
                if out_image.shape[2] == PATCH_SIZE + 1:
                    out_image = out_image[:, :, 1:]

                if out_image.shape[1] != PATCH_SIZE or out_image.shape[2] != PATCH_SIZE:
                    continue

                # Min-max scale the data to range [0, 1]
                out_image[out_image > maxValue] = maxValue
                out_image[out_image < minValue] = minValue
                out_image = (out_image - minValue) / (maxValue - minValue)

                out_meta = src.meta

                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                temp_tif = os.path.join(CROPPED_NDWI_MASK_DIR, f'{name}-ndwi_mask.tif')
                with rio.open(temp_tif, "w", **out_meta) as dest:
                    dest.write(out_image)

                print(f"Saved NDWI {temp_tif}")
            except Exception as e:
                print(f"Error processing NDWI tile {name}: {e}")


def export_sar_data(tiles, tif_file):
    if not os.path.exists(CROPPED_SAR_DIR):
        os.makedirs(CROPPED_SAR_DIR)

    with rio.open(tif_file) as src:
        # Read and flip the SAR image
        dataset_array = src.read()

        minValue = np.nanpercentile(dataset_array, 1)
        maxValue = np.nanpercentile(dataset_array, 99)

        # plot
        plt.imshow(dataset_array[0], cmap='gray')
        plt.show()

    # Process each tile
    for _, tile in tqdm(tiles.iterrows(), total=tiles.shape[0]):

        with rio.open(tif_file) as src:

            shape = [tile['geometry']]
            name = tile['id']
            print(f"Processing SAR tile: {name}")

            try:
                # Mask the flipped image for the current tile
                out_image, out_transform = rio.mask.mask(src, shape, crop=True)
                #out_image = out_image.astype(np.float32)

                if np.isnan(out_image).any():
                    print(f"Skipping SAR tile {name} due to NaN values.")
                    continue

                if out_image.shape[1] < PATCH_SIZE or out_image.shape[2] < PATCH_SIZE:
                    print(f"Skipping SAR tile {name} due to small size: {out_image.shape}")
                    continue

                if out_image.shape[1] == PATCH_SIZE + 1:
                    out_image = out_image[:, :-1, :]
                if out_image.shape[2] == PATCH_SIZE + 1:
                    out_image = out_image[:, :, 1:]

                if out_image.shape[1] != PATCH_SIZE or out_image.shape[2] != PATCH_SIZE:
                    continue

                # Min-max scale the data to range [0, 1]
                out_image[out_image > maxValue] = maxValue
                out_image[out_image < minValue] = minValue
                out_image = (out_image - minValue) / (maxValue - minValue)

                out_meta = src.meta

                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                temp_tif = os.path.join(CROPPED_SAR_DIR, f'{name}-sar_image.tif')
                with rio.open(temp_tif, "w", **out_meta) as dest:
                    dest.write(out_image)

                print(f"Saved SAR {temp_tif}")
            except Exception as e:
                print(f"Error processing SAR tile {name}: {e}")


def process_ndwi_files(ndwi_files):
    all_ndwi_tiles = []
    for ndwi_file in ndwi_files:
        print(f"Processing NDWI file: {ndwi_file}")
        ndwi_tiles = generate_tiles(ndwi_file, PATCH_SIZE)
        export_ndwi_mask_data(ndwi_tiles, ndwi_file)
        all_ndwi_tiles.append(ndwi_tiles)

    return pd.concat(all_ndwi_tiles, ignore_index=True)


def process_sar_files(sar_files, ndwi_tiles):
    for sar_file in sar_files:
        print(f"Processing SAR file: {sar_file}")
        export_sar_data(ndwi_tiles, sar_file)

def plot_tiles(ndwi_tile_path, sar_tile_path):
    with rio.open(ndwi_tile_path) as ndwi_src, rio.open(sar_tile_path) as sar_src:
        ndwi_image = ndwi_src.read(1)
        sar_image = sar_src.read(1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(ndwi_image, cmap='gray')
        axes[0].set_title('NDWI Tile')
        axes[1].imshow(sar_image, cmap='gray')
        axes[1].set_title('SAR Tile')
        plt.show()


def main():
    ndwi_files = sorted(glob.glob(os.path.join(NDWI_MASK_DIR, '*.tif')))
    sar_files = sorted(glob.glob(os.path.join(SAR_DIR, '*.tif')))

    ndwi_tiles = process_ndwi_files(ndwi_files)
    process_sar_files(sar_files, ndwi_tiles)

    # Plot the NDWI and SAR tile for tile 16288
    plot_tiles(
        os.path.join(CROPPED_NDWI_MASK_DIR, 'tile-16288-ndwi_mask.tif'),
        os.path.join(CROPPED_SAR_DIR, 'tile-16288-sar_image.tif')
    )


if __name__ == "__main__":
    main()
