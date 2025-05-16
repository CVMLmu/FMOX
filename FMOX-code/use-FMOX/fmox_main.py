import os
import sys
import json
from pathlib import Path
import pandas as pd

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
import size_label_count
import efficientam_evaluation


def main():
    # ----------------------------------------------
    # File Paths
    # ---------------------------------------------
    data_folder = "../Original_Dataset/"
    fmox_json_path = "../FMOX-Jsons/FMOX_All4.json"  # fmox json annotations
    fmox_csv_path = "./FMOX_All4_statistics.csv"
    efficienttam_json_path = "../EfficientTAM-Jsons/efficienttam_All4.json"
    averageTIoU_path = "./EfficientTAM_averageTIoU.csv"

    # ----------------------------------------------
    # Check if dataset is not downloaded - download
    # ---------------------------------------------
    # """ The dataset is downloaded and extracted into a folder named "Original_Dataset". """
    # data_path = Path('../Original_Dataset')
    # if not data_path.exists():  # Check if the directory exists
    #     download_datasets.download_unzip_data(data_path)
    # else:
    #     # Check if the directory has files
    #     if list(data_path.iterdir()):  # folder is not empty, so pass
    #         pass
    #     else:
    #         download_datasets.download_unzip_data(data_path)
    #
    # """Change "FMOv2/FMOv2" to "FMOv2/imgs" because other dataset has "imgs" - to run the code """
    # current_folder_name = "../Original_Dataset/FMOv2/FMOv2"
    # new_folder_name = "../Original_Dataset/FMOv2/imgs"
    # try:
    #     os.rename(current_folder_name, new_folder_name)
    #     print(f"Folder renamed from '{current_folder_name}' to '{new_folder_name}'")
    # except FileNotFoundError:
    #     print(f"Error: The folder '{current_folder_name}' does not exist.")
    # except FileExistsError:
    #     print(f"Error: The folder '{new_folder_name}' already exists.")
    # except Exception as e:
    #     print(f"An error occurred: {e}")

    # ----------------------------------------------------------
    # Access Bounding Boxes from a JSON Annotation File
    # ----------------------------------------------------------
    # access_json_bboxes.access_bboxes(data_folder, fmox_json_path)

    # ----------------------------------------------------------
    # Evaluate EfficientTAM JSON results with FMOX JSON
    # ----------------------------------------------------------
    # efficientam_evaluation.evaluate_efficienttam(data_folder, fmox_json_path, efficienttam_json_path, averageTIoU_path)

    # ----------------------------------------------------------
    # Create CSV from FMOX All4 JSON
    # ----------------------------------------------------------
    """ Create csv file to save information e.g. "Main Dataset", "Subsequence", "Total Frame Number",
    "FMO Exists Frame Number", "Average Object Size", "Object Size Levels" for table and graphs.
    Get the number of object levels for each sequence e.g.
    {’extremely_tiny’:1,’large’:2,’medium’: 22,’small’:5,’tiny’:8} 
    Note: For tbd whole image sequence need to run """
    # FMOX_all4_json_to_CSV.json_to_csv(fmox_json_path, fmox_csv_path)

    # ----------------------------------------------------------
    #  CSV to graphics: Bar plot, Scatter plot, Box plot etc.
    # ----------------------------------------------------------
    """ Scatter plot for comparing "Number of FMO exist frame" vs. "Number of total frame" """
    # fmox_csv_data = pd.read_csv(fmox_csv_path)
    # csv_to_graphics.cvs_viz2(fmox_csv_data)

    """Stacked bar chart for visualization object size levels for each dataset and subsequences."""
    # csv_to_graphics.visualize_object_size_levels(fmox_csv_data)


if __name__ == "__main__":
    main()

