import os
import requests
import zipfile

# ------------------------------------------------------------------------
# download datasets from the links  note: matlab files is not downloading
# ------------------------------------------------------------------------

# Falling objects dataset as a set of PNG images both for high-speed (ground truth) and
# low-speed versions is available here (together with ground truth annotations).
falling_imgs_gt = "http://ptak.felk.cvut.cz/personal/rozumden/falling_imgs_gt.zip"

# The whole FMOv2 dataset (decomposed to png images) is available here.
fmov2 = "http://ptak.felk.cvut.cz/personal/rozumden/FMOv2.zip"
# Separate ground truth is available the FMO data set (MATLAB, text) and the FMOv2 data
# set (MATLAB, text). Ground truth in png format for the whole FMOv2 dataset is here.
fmov2_gt_rle_txt = "http://cmp.felk.cvut.cz/fmo/files/gt-fmov2-txt-2017-05-26.zip"
fmov2_gt_png = "http://ptak.felk.cvut.cz/personal/rozumden/FMOv2_gt.zip"

# TbD with ground truth can be downloaded here, and as PNG images here.
tbd_gt = "http://ptak.felk.cvut.cz/personal/rozumden/TbD.zip"
tbd = "http://ptak.felk.cvut.cz/personal/rozumden/TbD_benchmark.zip"

# TbD-3D dataset with ground truth in Matlab format can be downloaded here.
# Dataset as a set of PNG images both for high-speed (ground truth) and low-speed versions is available here.
tbd3d_imgs = "http://ptak.felk.cvut.cz/personal/rozumden/TbD-3D-imgs.zip"

urls = {
    "Falling_Object" : [falling_imgs_gt],
    "FMOv2": [fmov2, fmov2_gt_rle_txt, fmov2_gt_png],
    "TbD": [tbd, tbd_gt],
    "TbD-3D": [tbd3d_imgs]
}

def download_unzip_data(data_path):
    os.makedirs(data_path, exist_ok=True)   # Create the directory if do not exist

    # Download and unzip each file
    for folder_name, file_urls in urls.items():
        # Create a directory for each folder name inside the extract_dir
        folder_extract_path = os.path.join(data_path, folder_name)
        os.makedirs(folder_extract_path, exist_ok=True)

        for url in file_urls:
            # Extract the filename from the URL
            filename = url.split('/')[-1]
            local_filename = os.path.join(data_path, filename)

            # Check if the zip file already exists
            if os.path.exists(local_filename):
                print(f"Zip file already exists, skipping download: {local_filename}")
                continue

            try:
                # Download the file
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(local_filename, 'wb') as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)
                    print(f"File downloaded successfully: {local_filename}")
                else:
                    print(f"Failed to download file {filename}. Status code: {response.status_code}")
                    continue

                # Check if the file was downloaded
                if not os.path.exists(local_filename):
                    print(f"Downloaded file does not exist: {local_filename}")
                    continue

                # Unzip the file
                with zipfile.ZipFile(local_filename, 'r') as zip_ref:
                    # Extract all files directly into the folder_extract_path
                    zip_ref.extractall(folder_extract_path)
                    print(f"Extracted files to: {folder_extract_path}")

                # Check if extraction was successful
                if not os.path.exists(folder_extract_path) or not os.listdir(folder_extract_path):
                    print(f"Extraction failed or no files found in: {folder_extract_path}")
                    continue

                # Remove the downloaded zip file (optional)
                os.remove(local_filename)
                print(f"Downloaded file removed: {local_filename}")

            except requests.exceptions.RequestException as e:
                print(f"An error occurred while downloading the file {filename}: {e}")
            except zipfile.BadZipFile as e:
                print(f"An error occurred while unzipping the file {filename}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred with the file {filename}: {e}")


# Define the directories for downloading and extracting files
data_path = './Original_Dataset'
if os.listdir(data_path):  # folder has files, so pass
    pass
else:
    download_unzip_data(data_path)
