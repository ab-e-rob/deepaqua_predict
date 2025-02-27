import os
import zipfile
import shutil

wetland_name = 'Upper_lough_erne'

# Directory containing your zip files
zip_dir = 'gee_zipped'

# Destination folder
destination_folder = 'sar_imagery/' + wetland_name + '_sar'

# List all zip files in the directory
zip_files = [f for f in os.listdir(zip_dir) if f.endswith('.zip')]

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Extract and combine the contents of each zip file into the destination folder
for zip_file in zip_files:
    with zipfile.ZipFile(os.path.join(zip_dir, zip_file), 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

# Move all the extracted files to the root of the destination folder
for root, _, files in os.walk(destination_folder):
    for file in files:
        source_path = os.path.join(root, file)
        destination_path = os.path.join(destination_folder, file)
        shutil.move(source_path, destination_path)

print("Combination complete.")
