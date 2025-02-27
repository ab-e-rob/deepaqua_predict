import time
import geetools
from geetools import batch
import ee

### Define parameters ###
area_name = 'Upper_lough_erne'
ORBIT_PASS = 'DESCENDING'
START_DATE = '2018-01-01'
END_DATE = '2018-12-31'

# For monthly composite export only
START_YEAR = 2020
END_YEAR = 2021

# Define your bounding box for areas of interest
def get_area_of_interest(area_name):
    """
    Define the area of interest based on the provided area name.

    Args:
        area_name (str): Name of the area of interest.

    Returns:
        ee.Geometry: The bounding box geometry for the specified area.
    """
    areas_of_interest = {
        'Vasikkavouma': ee.Geometry.Polygon(
            [[[23.11432701111463, 67.19432612267958], [23.28644369047891, 67.19432612267958],
              [23.28644369047891, 67.25682392423104], [23.11432701111463, 67.25682392423104]]]),
        'Tavvavuoma': ee.Geometry.Polygon(
            [[[19.937500000000032, 68.39583333333334], [21.050000000000022, 68.39583333333334],
              [21.050000000000022, 68.70890062120226], [19.937500000000032, 68.70890062120226]]]),
        'Helge': ee.Geometry.Polygon(
            [[[14.068034007763288, 55.85561234963029], [14.265849302303536, 55.85561234963029],
              [14.265849302303536, 56.1001022452225], [14.068034007763288, 56.1001022452225]]]),
        'Osten': ee.Geometry.Polygon(
            [[[13.873464893490109, 58.534126363341045], [13.96474439943455, 58.534126363341045],
              [13.96474439943455, 58.59721449015666], [13.873464893490109, 58.59721449015666], ]]),
        'Persofjarden': ee.Geometry.Polygon(
            [[[21.94304304010427, 65.71885538936476], [22.171295698637607, 65.71885538936476],
              [22.171295698637607, 65.83352842819448], [21.94304304010427, 65.83352842819448], ]]),
        'Takern': ee.Geometry.Polygon(
            [[[14.707542295140854, 58.319238866980186], [14.921170146185375, 58.319238866980186],
              [14.921170146185375, 58.387983447078454], [14.707542295140854, 58.387983447078454], ]]),
        'Farnebofjarden': ee.Geometry.Polygon(
            [[[16.64640602771607, 60.0337779207145], [17.018837808717933, 60.0337779207145],
              [17.018837808717933, 60.33959057664384], [16.64640602771607, 60.33959057664384], ]]),
        'Kulbacksliden': ee.Geometry.Polygon(
            [[[19.509302456000057, 64.15199152000008], [19.593295690000048, 64.15199152000008],
              [19.593295690000048, 64.19990718800005], [19.509302456000057, 64.19990718800005],
              ]]),
        'Upper_lough_erne': ee.Geometry.Polygon(
            [[[-7.632520304999957, 54.12009783600007], [-7.269607505999943, 54.12009783600007],
              [-7.269607505999943, 54.298014841000054], [-7.632520304999957, 54.298014841000054],
              ]])

    }


    return areas_of_interest[area_name]

# Downloads all SAR images for the AOI between the start and end dates
def bulk_export_sar(area_name, include_diff=False):
    """
    Export all SAR images for the given area of interest (AOI) within the specified date range.

    Args:
        area_name (str): Name of the area of interest.
        include_diff (bool): Whether to include the VV-VH difference band.
    """
    start_date = START_DATE
    end_date = END_DATE

    aoi = get_area_of_interest(area_name)

    collection = ee.ImageCollection('COPERNICUS/S1_GRD')\
        .filterDate(start_date, end_date)\
        .filterBounds(aoi)\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
        .filter(ee.Filter.eq('instrumentMode', 'IW'))\
        .filter(ee.Filter.eq('orbitProperties_pass', ORBIT_PASS))\
        .filter(ee.Filter.eq('resolution', 'H'))\
        .filter(ee.Filter.eq('resolution_meters', 10))\
        .map(lambda img: img.toFloat())  # Ensure all bands are Float32

    print('SAR Collection size:', collection.size().getInfo())

    # Function to export each image in the collection
    def export_image(img):
        """
        Export a single image to Google Drive.

        Args:
            img (ee.Image): The image to export.
        """
        img_id = img.get('system:index').getInfo()  # Get the image ID
        export_description = f'{img_id}'  # Custom name using the image ID

        # Include VV-VH difference band if requested
        if include_diff:
            vv_minus_vh = img.select('VV').subtract(img.select('VH')).rename('VV_VH_diff')
            img = img.addBands(vv_minus_vh)
            bands = ['VV', 'VH', 'VV_VH_diff']
        else:
            bands = ['VV', 'VH']

        task = ee.batch.Export.image.toDrive(
            image=img.select(bands),
            description=export_description,
            folder=f'{area_name}_sar_export',  # Google Drive folder name
            scale=10,
            region=aoi,
            crs='EPSG:4326',
            maxPixels=1e12
        )
        task.start()

    # Export each image in the collection
    image_list = collection.toList(collection.size())
    for i in range(collection.size().getInfo()):
        img = ee.Image(image_list.get(i))
        export_image(img)

# Downloads monthly composite SAR images for the AOI between the start and end years
def bulk_export_monthly_sar(area_name):
    """
    Export monthly composite SAR images for the given area of interest (AOI) between the specified years.

    Args:
        area_name (str): Name of the area of interest.
    """
    start_year = START_YEAR
    end_year = END_YEAR

    aoi = get_area_of_interest(area_name)

    collection = ee.ImageCollection('COPERNICUS/S1_GRD')\
        .filterDate(f'{start_year}-01-01', f'{end_year}-12-31')\
        .filterBounds(aoi)\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
        .filter(ee.Filter.eq('instrumentMode', 'IW'))\
        .filter(ee.Filter.eq('orbitProperties_pass', ORBIT_PASS))\
        .filter(ee.Filter.eq('resolution', 'H'))\
        .filter(ee.Filter.eq('resolution_meters', 10))

    print('SAR Collection size:', collection.size().getInfo())

    # Create a function to calculate the median of an ImageCollection and set the date property
    def yearly_monthly_composite(image_collection):
        """
        Create monthly composites from the image collection.

        Args:
            image_collection (ee.ImageCollection): The collection of images to process.

        Returns:
            ee.ImageCollection: A collection of monthly composite images.
        """
        def process_year_month(y, m):
            start = ee.Date.fromYMD(y, m, 1)
            end = start.advance(1, 'month')
            composite = image_collection.filterDate(start, end).median().set('system:time_start', start.millis())
            return composite.set('date', start.format('YYYY-MM-dd'))

        # Create a list of years from start_year to end_year
        years = ee.List.sequence(start_year, end_year)

        # Create a list of months from 1 to 12
        months = ee.List.sequence(1, 12)

        # Map over the years and months and apply the process_year_month function
        return ee.ImageCollection.fromImages(
            years.map(lambda y: months.map(lambda m: process_year_month(y, m))).flatten()
        )

    # Apply the yearly_monthly_composite function to create yearly monthly composites
    yearly_monthly_composites = yearly_monthly_composite(collection)

    # Batch export to Google Drive
    geetools.batch.Export.imagecollection.toDrive(
        yearly_monthly_composites,
        f'{area_name}_monthly_sar',
        scale=10,
        dataType="float",
        region=aoi,
        crs='EPSG:4326',
        datePattern='YYYY-MM-dd',
        extra=None,
        verbose=False
    )

def main():
    """
    Main function to set up Google Earth Engine and run the export functions.
    """
    # Set up Google Earth Engine
    ee.Authenticate()
    ee.Initialize()

    # Choose which function to run
    bulk_export_sar(area_name, include_diff=True)
    #bulk_export_monthly_sar(area_name)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    total_time = end - start
    print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
