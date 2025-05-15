import os
import sys
import json
from pathlib import Path

script_name = 'download_datasets'
# Define the path to the parent script
script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', f'{script_name}.py'))
# Check if the script exists
if not os.path.isfile(script_path):
    print(f"Script not found: {script_path}")
else:
    # Add the parent directory to the Python path
    parent_dir = os.path.dirname(script_path)
    sys.path.append(parent_dir)
    try:
        import download_datasets
        print("Module imported successfully.")
    except ImportError as e:
        print(f"Error importing module: {e}")


# import download_datasets
import FMOX_all4_json_to_CSV
import access_json_bboxes
import csv_to_graphics


def main():

    # ----------------------------------------------
    # Check if dataset is not downloaded - download
    # ---------------------------------------------
    """ The dataset is downloaded and extracted into a folder named "Original_Dataset". """
    data_path = Path('../Original_Dataset')
    if not data_path.exists():  # Check if the directory exists
        download_datasets.download_unzip_data(data_path)
    else:
        # Check if the directory has files
        if list(data_path.iterdir()):  # folder is not empty, so pass
            pass
        else:
            download_datasets.download_unzip_data(data_path)

    # ----------------------------------------------------------
    # Access Bounding Boxes from a JSON Annotation File
    # ----------------------------------------------------------
    # access_json_bboxes.access_bboxes()

    # ----------------------------------------------------------
    # Create CSV from FMOX All4 JSON
    # ----------------------------------------------------------
    # Create csv file to save informations e.g. "Main Dataset", "Subsequence", "Total Frame Number",
    # "FMO Exists Frame Number", "Average Object Size", "Object Size Levels" for table and graphs.
    fmox_json_path = "../FMOX-Jsons/FMOX_All4.json"
    fmox_csv_path = "./FMOX_All4_statistics.csv"
    FMOX_all4_json_to_CSV.json_to_csv(fmox_json_path, fmox_csv_path)

    # ----------------------------------------------------------
    #  CSV to graphics
    # ----------------------------------------------------------
    # Different kind of visualization functions : Bar plot, Scatter plot, Box plot etc.
    #  Scatter plot for comparing "Number of FMO exist frame" vs. "Number of total frame"
    # csv_to_graphics.cvs_viz2()
    # Stacked bar chart for visualization object size levels for each dataset and subsequences.
    # csv_to_graphics.visualize_object_size_levels()

    # ----------------------------------------------------------
    # Size Label Count
    # ----------------------------------------------------------


    # ----------------------------------------------------------
    # Evaluate EfficientTAM JSON results with FMOX JSON
    # ----------------------------------------------------------



    # ----------------------------------------------------------
    # ----------------------------------------------------------



if __name__ == "__main__":
    main()