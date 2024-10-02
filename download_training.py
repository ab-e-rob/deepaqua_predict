import ee
import geopandas as gpd

### Set parameters
START_DATE = '2023-08-01'
END_DATE = '2024-09-01'
PATH_TO_SHP = 'training_data/orebro_lan.shp'


def get_matching_dates_from_shapefile(shapefile_path, start_date, end_date):
    """
    Get a list of matching dates between Sentinel-2 and Sentinel-1 imagery
    with 0% cloud cover between a start date and an end date within the frame of a shapefile.

    Args:
        shapefile_path (str): The file path to the shapefile.
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.

    Returns:
        list: List of matching dates.
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Get the bounding box of all geometries
    bbox = gdf.total_bounds  # returns (minx, miny, maxx, maxy)
    aoi = ee.Geometry.Rectangle([bbox[0], bbox[1], bbox[2], bbox[3]])

    # Define the date range
    start_date = ee.Date(start_date)
    end_date = ee.Date(end_date)

    # Get Sentinel-2 image collection with 0% cloud cover
    s2_collection = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 0))

    # Get unique dates for Sentinel-2
    s2_dates = s2_collection.aggregate_array('system:time_start') \
        .map(lambda d: ee.Date(d).format('YYYY-MM-dd')) \
        .distinct()

    # Convert to Python list
    s2_dates = s2_dates.getInfo()

    # Initialize empty list for matching dates
    matching_dates = []

    # Loop through Sentinel-2 dates to find matching Sentinel-1 dates
    for date in s2_dates:
        # Convert the date to ee.Date
        date_ee = ee.Date(date)

        # Get the Sentinel-2 image for the date
        s2_image = s2_collection.filterDate(date_ee, date_ee.advance(1, 'day')).first()

        if s2_image is None:
            continue

        # Get the bounding box of the Sentinel-2 image
        s2_bbox_info = s2_image.geometry().bounds().getInfo()
        s2_bbox = s2_bbox_info['coordinates'][0]
        if len(s2_bbox) == 5:
            s2_bbox = [s2_bbox[0][0], s2_bbox[0][1], s2_bbox[2][0], s2_bbox[2][1]]
            s2_bbox = ee.Geometry.Rectangle(s2_bbox)
        else:
            continue

        # Get the Sentinel-1 image collection for the date within the Sentinel-2 bbox
        s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(s2_bbox) \
            .filterDate(date_ee, date_ee.advance(1, 'day'))

        # Get unique dates for Sentinel-1
        s1_dates = s1_collection.aggregate_array('system:time_start') \
            .map(lambda d: ee.Date(d).format('YYYY-MM-dd')) \
            .distinct()

        # Convert to Python list
        s1_dates = s1_dates.getInfo()

        # Find the intersection of the two lists
        common_dates = list(set([date]) & set(s1_dates))
        if common_dates:
            matching_dates.extend(common_dates)

    matching_dates = list(set(matching_dates))
    matching_dates.sort()

    print("Matching dates between Sentinel-2 and Sentinel-1 with 0% cloud cover:")
    for i, date in enumerate(matching_dates):
        print(f"{i + 1}: {date}")

    return matching_dates


def download_ndwi_and_s1(shapefile_path, date):
    """
    Download the NDWI mask from Sentinel-2 imagery and the corresponding Sentinel-1 image for a specific date.

    Args:
        shapefile_path (str): The file path to the shapefile.
        date (str): The date in the format 'YYYY-MM-DD'.

    Returns:
        None
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Get the bounding box of all geometries
    bbox = gdf.total_bounds  # returns (minx, miny, maxx, maxy)
    aoi = ee.Geometry.Rectangle([bbox[0], bbox[1], bbox[2], bbox[3]])

    # Define the date
    date = ee.Date(date)
    start_date = date
    end_date = date.advance(1, 'day')

    # Get the Sentinel-2 image for the date with 0% cloud cover
    s2_image = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 0)) \
        .first()

    if s2_image is None or s2_image.bandNames().size().getInfo() < 1:
        print("No Sentinel-2 image found for the specified date.")
        return

    # Get the bounding box of the Sentinel-2 image
    s2_bbox_info = s2_image.geometry().bounds().getInfo()
    s2_bbox = s2_bbox_info['coordinates'][0]

    # Ensure the correct format for ee.Geometry.Rectangle
    if len(s2_bbox) == 5:
        s2_bbox = [s2_bbox[0][0], s2_bbox[0][1], s2_bbox[2][0], s2_bbox[2][1]]
        s2_bbox = ee.Geometry.Rectangle(s2_bbox)
    else:
        raise ValueError("Bounding box does not have 4 coordinates")

    # Get the Sentinel-1 image for the date within the Sentinel-2 bbox and cast to float32
    s1_image = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(s2_bbox) \
        .filterDate(start_date, end_date) \
        .first() \
        .select('VV', 'VH') \
        .toFloat()

    if s1_image is None or s1_image.bandNames().size().getInfo() < 1:
        print("No Sentinel-1 image found for the specified date.")
        return

    # Extract image IDs
    s1_image_id = s1_image.id().getInfo()
    s2_image_id = s2_image.id().getInfo()

    # Compute the NDWI mask and cast to float32
    ndwi = s2_image.normalizedDifference(['B3', 'B8']).rename('NDWI').toFloat()

    # Create a mask for the NDWI
    ndwi_mask = ndwi.gte(0)

    # Define export parameters
    export_params = {
        'scale': 10,
        'region': s2_bbox,  # Use the bounding box of the Sentinel-2 image
        'fileFormat': 'GeoTIFF',
        'maxPixels': 1e13,
        'folder': 'training_data',
    }

    # Export the NDWI mask
    ndwi_task = ee.batch.Export.image.toDrive(
        image=ndwi_mask,
        description='NDWI_Mask_{}'.format(s2_image_id),
        **export_params
    )
    ndwi_task.start()

    # Export the Sentinel-1 image
    s1_task = ee.batch.Export.image.toDrive(
        image=s1_image,
        description='Sentinel-1_{}'.format(s1_image_id),
        **export_params
    )
    s1_task.start()

    print('Export tasks started. Check your Google Drive for the results.')


### Full cycle
def full_cycle():
    """
    Run the full cycle of downloading Sentinel-2 and Sentinel-1 imagery, processing it, and training a model.
    """
    # Set up Google Earth Engine
    ee.Initialize()

    # Get the matching dates
    matching_dates = get_matching_dates_from_shapefile(PATH_TO_SHP, START_DATE, END_DATE)

    # Prompt user to select a date
    date_index = int(input(f"Select a date (1-{len(matching_dates)}): ")) - 1
    if 0 <= date_index < len(matching_dates):
        selected_date = matching_dates[date_index]
        download_ndwi_and_s1(PATH_TO_SHP, selected_date)
    else:
        print("Invalid selection. Please try again.")

    print("Full cycle completed successfully.")


def main():
    full_cycle()


if __name__ == "__main__":
    main()
