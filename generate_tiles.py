import os
import numpy as np
import rasterio as rio
import rasterio.mask
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

### Parameters

NDWI_MASK_DIR = 'training_data/ndwi_mask/NDWI_Mask_2018-07-04.tif'
CROPPED_NDWI_MASK_DIR = 'training_data/cropped_ndwi_mask'
SAR_DIR = 'training_data/sar/SAR_Image_2018-07-04.tif'
CROPPED_SAR_DIR = 'training_data/cropped_sar'
PATCH_SIZE = 64
EPSG = 32633


## Generate tiles for the entire NDWI image

def generate_tiles(image_file, output_file, size):
    """Generates size x size polygon tiles and saves as GeoJSON."""
    with rio.open(image_file) as raster:
        width, height = raster.width, raster.height
        geo_dict = {'id': [], 'geometry': [], 'area': []}

        for w in tqdm(range(0, width, size), total=width // size):
            for h in range(0, height, size):
                window = rio.windows.Window(w, h, size, size)
                bbox = rio.windows.bounds(window, raster.transform)
                geo_dict['id'].append(f'tile-{len(geo_dict["id"])}')
                geo_dict['area'].append('Entire_NDWI_Image')
                geo_dict['geometry'].append(box(*bbox))

        results = gpd.GeoDataFrame(pd.DataFrame(geo_dict), crs=f"epsg:{EPSG}")
        results.to_file(output_file, driver="GeoJSON")
        print(f"Generated {len(results)} NDWI tiles.")
        return results


def export_ndwi_mask_data(tiles, tif_file):
    # Ensure the output directory exists
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
                with rio.open(tif_file) as src:
                    out_image, out_transform = rio.mask.mask(src, shape, crop=True)

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

                    # temp_png = os.path.join(CROPPED_NDWI_MASK_DIR, f'{name}-ndwi_mask.png')
                    # cm = plt.get_cmap(ListedColormap(["black", "cyan"]))
                    # colored_image = cm(out_image[0])
                    # Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(temp_png)

                    print(f"Saved NDWI {temp_tif}")
            except Exception as e:
                print(f"Error processing NDWI tile {name}: {e}")


def full_cycle_ndwi(tif_file):
    tiles = generate_tiles(tif_file, 'ndwi_tiles.geojson', PATCH_SIZE)
    print(f"Generated {len(tiles)} tiles for the entire NDWI image.")
    export_ndwi_mask_data(tiles, tif_file)
    print("Full cycle for NDWI image completed successfully.")


## Generate tiles for the entire SAR image

def generate_sar_tiles(image_file, output_file, size):
    """Generates size x size polygon tiles and saves as GeoJSON."""
    with rio.open(image_file) as raster:
        width, height = raster.width, raster.height
        geo_dict = {'id': [], 'geometry': [], 'area': []}

        for w in tqdm(range(0, width, size), total=width // size):
            for h in range(0, height, size):
                window = rio.windows.Window(w, h, size, size)
                bbox = rio.windows.bounds(window, raster.transform)
                geo_dict['id'].append(f'tile-{len(geo_dict["id"])}')
                geo_dict['area'].append('Entire_SAR_Image')
                geo_dict['geometry'].append(box(*bbox))

        results = gpd.GeoDataFrame(pd.DataFrame(geo_dict), crs=f"epsg:{EPSG}")
        results.to_file(output_file, driver="GeoJSON")
        print(f"Generated {len(results)} SAR tiles.")
        return results


def export_sar_data(tiles, tif_file):
    # Ensure the output directory exists
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
                with rio.open(tif_file) as src:
                    out_image, out_transform = rio.mask.mask(src, shape, crop=True)

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

                    # temp_png = os.path.join(CROPPED_SAR_DIR, f'{name}-sar_image.png')
                    # cm = plt.get_cmap(ListedColormap(["black", "gray"]))
                    # colored_image = cm(out_image[0])
                    # Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(temp_png)

                    print(f"Saved SAR {temp_tif}")
            except Exception as e:
                print(f"Error processing SAR tile {name}: {e}")


def full_cycle_sar(tif_file):
    tiles = generate_sar_tiles(tif_file, 'sar_tiles.geojson', PATCH_SIZE)
    print(f"Generated {len(tiles)} tiles for the entire SAR image.")
    export_sar_data(tiles, tif_file)
    print("Full cycle for SAR image completed successfully.")


def main():
    # Uncomment the desired function call to process NDWI or SAR data
    full_cycle_ndwi(NDWI_MASK_DIR)
    full_cycle_sar(SAR_DIR)


if __name__ == "__main__":
    main()
