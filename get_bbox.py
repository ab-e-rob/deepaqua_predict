import geopandas as gpd
from shapely.geometry import Polygon, mapping
import json
import matplotlib.pyplot as plt

### Define wetland/wetlandscape name
STUDY_AREA = 'Upper_lough_erne'

### Define shapefile path ###
SHAPEFILE_PATH = f'study_areas/{STUDY_AREA}.shp'

def calculate_bounding_box(shapefile_path):
    """
    Calculate the bounding box of a shapefile and convert it to the specified format.

    Args:
        shapefile_path (str): Path to the shapefile.

    Returns:
        dict: Bounding box in the specified format.
    """
    # Load the shapefile using GeoPandas
    gdf = gpd.read_file(shapefile_path)

    # Calculate the bounding box
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]

    # Create a bounding box polygon
    bounding_box = Polygon([
        [bounds[0], bounds[1]],  # (minx, miny)
        [bounds[2], bounds[1]],  # (maxx, miny)
        [bounds[2], bounds[3]],  # (maxx, maxy)
        [bounds[0], bounds[3]]   # (minx, maxy)
    ])

    # Convert to the specified format
    bounding_box_geojson = mapping(bounding_box)
    coordinates = bounding_box_geojson['coordinates'][0]

    formatted_bounding_box = {
        STUDY_AREA: f"ee.Geometry.Polygon({json.dumps([coordinates])})"
    }


    # Plot the bounding box using matplotlib
    fig, ax = plt.subplots()
    gdf.plot(ax=ax, color='blue', alpha=0.5)
    x, y = bounding_box.exterior.xy
    ax.plot(x, y, color='red')
    plt.title(f'Bounding Box for {STUDY_AREA}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


    return formatted_bounding_box


bounding_box = calculate_bounding_box(SHAPEFILE_PATH)
print(bounding_box)






