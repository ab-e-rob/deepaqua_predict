import os
import numpy as np
import rasterio as rio
import rasterio.mask
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import glob
from rasterio.warp import calculate_default_transform, reproject, Resampling

### Parameters

NDWI_MASK_DIR = 'training_data/ndwi_mask'
CROPPED_NDWI_MASK_DIR = 'training_data/ndwi_tiles'
SAR_DIR = 'training_data/sar'
CROPPED_SAR_DIR = 'training_data/sar_tiles'
PATCH_SIZE = 64
EPSG = 32633

tile_id_counter = 0  # Global tile ID counter


def resample_image(src_path, dst_path, match_path):
    """Resample image to match the geotransform of another image."""
    with rio.open(src_path) as src:
        with rio.open(match_path) as match:
            transform, width, height = calculate_default_transform(
                src.crs, match.crs, match.width, match.height, *match.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': match.crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rio.open(dst_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rio.band(src, i),
                        destination=rio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=match.crs,
                        resampling=Resampling.nearest)


def check_alignment(ndwi_file, sar_file):
    """Check that NDWI and SAR images have the same alignment and resolution."""
    with rio.open(ndwi_file) as ndwi_src, rio.open(sar_file) as sar_src:
        if ndwi_src.transform != sar_src.transform:
            print(f"Resampling SAR image to match NDWI image.")
            resampled_sar_file = sar_file.replace('.tif', '_resampled.tif')
            resample_image(sar_file, resampled_sar_file, ndwi_file)
            return resampled_sar_file
        if ndwi_src.crs != sar_src.crs:
            raise ValueError("NDWI and SAR images do not have the same CRS.")
        if ndwi_src.width != sar_src.width or ndwi_src.height != sar_src.height:
            raise ValueError("NDWI and SAR images do not have the same dimensions.")
    return sar_file


def generate_tiles(image_file, size):
    """Generates size x size polygon tiles."""
    global tile_id_counter
    with rio.open(image_file) as raster:
        width, height = raster.width, raster.height
        geo_dict = {'id': [], 'geometry': []}

        for w in tqdm(range(0, width, size), total=width // size):
            for h in range(0, height, size):
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
        dataset_array = src.read()
        min_value = np.nanpercentile(dataset_array, 1)
        max_value = np.nanpercentile(dataset_array, 99)

        for _, tile in tqdm(tiles.iterrows(), total=tiles.shape[0]):
            shape = [tile['geometry']]
            name = tile['id']
            print(f"Processing NDWI tile: {name}")

            try:
                out_image, out_transform = rio.mask.mask(src, shape, crop=True)
                out_image = out_image.astype(np.float32)

                if np.isnan(out_image).any():
                    print(f"Skipping NDWI tile {name} due to NaN values.")
                    continue

                if out_image.shape[1] < PATCH_SIZE or out_image.shape[2] < PATCH_SIZE:
                    print(f"Skipping NDWI tile {name} due to small size: {out_image.shape}")
                    continue

                out_image = out_image[:, :PATCH_SIZE, :PATCH_SIZE]
                out_image = np.clip(out_image, min_value, max_value)
                out_image = (out_image - min_value) / (max_value - min_value)

                out_meta = src.meta.copy()
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
        dataset_array = src.read()
        min_value = np.nanpercentile(dataset_array, 1)
        max_value = np.nanpercentile(dataset_array, 99)

        for _, tile in tqdm(tiles.iterrows(), total=tiles.shape[0]):
            shape = [tile['geometry']]
            name = tile['id']
            print(f"Processing SAR tile: {name}")

            try:
                out_image, out_transform = rio.mask.mask(src, shape, crop=True)
                out_image = out_image.astype(np.float32)

                if np.isnan(out_image).any():
                    print(f"Skipping SAR tile {name} due to NaN values.")
                    continue

                if out_image.shape[1] < PATCH_SIZE or out_image.shape[2] < PATCH_SIZE:
                    print(f"Skipping SAR tile {name} due to small size: {out_image.shape}")
                    continue

                out_image = out_image[:, :PATCH_SIZE, :PATCH_SIZE]
                out_image = np.clip(out_image, min_value, max_value)
                out_image = (out_image - min_value) / (max_value - min_value)

                out_meta = src.meta.copy()
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


def main():
    ndwi_files = sorted(glob.glob(os.path.join(NDWI_MASK_DIR, '*.tif')))
    sar_files = sorted(glob.glob(os.path.join(SAR_DIR, '*.tif')))

    resampled_sar_files = []
    for ndwi_file, sar_file in zip(ndwi_files, sar_files):
        resampled_sar_file = check_alignment(ndwi_file, sar_file)
        resampled_sar_files.append(resampled_sar_file)

    ndwi_tiles = process_ndwi_files(ndwi_files)
    process_sar_files(resampled_sar_files, ndwi_tiles)


if __name__ == "__main__":
    main()
