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
import create_fmov2_json
import create_tbd_json
import dataset_loader.create_json_via_benchmark_loader as fmo_data_loader
import rle_to_seg_mask_img
import combine_all_mask_to_single_img
import tbd_visualize_bboxes


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

    # Ensure the output directory exists
    fmox_json_path = "../FMOX-Jsons"
    os.makedirs(os.path.dirname(fmox_json_path), exist_ok=True)

    # ----------------------------------------------------------
    # Create FMOv2 JSON Annotation File
    # ----------------------------------------------------------
    """ Contour detection is applied to the segmentation mask images to obtain bounding boxes. 
     While obtaining the bounding boxes, object size labels are calculated and saved in a JSON file. """

    create_fmov2_json.get_fmov2_json()

    # ----------------------------------------------------------
    # Create TbD JSON Annotation File
    # ----------------------------------------------------------
    """ Ground-truth trajectory text files (gt.txt) used to obtain the bounding boxes for whole sequence of frames. 
     To obtain only FMO bounding boxes; please use code which provided for Falling Object, TbD-3D Datasets. 
     While obtaining the bounding boxes, object size labels are calculated and saved in a JSON file. """

    # create_tbd_json.get_tbd_json()

    # ---------------------------------------------------------------------
    # Create JSON Annotation File For Falling Object, TbD-3D Datasets
    # ---------------------------------------------------------------------
    """ - Acknowledgments: The code is provided (in "dataset_loader" folder) adapted 
    from the [fmo-deblurring-benchmark](https://github.com/rozumden/fmo-deblurring-benchmark) and
    some modifications made to fit our specific use case. For more details, please visit the repository.
    - However if you want to regenerate it, please refer below file:"""

    # fmo_data_loader.create_json()

    # ----------------------------------------------------------
    # Combine JSONS - create FMOX_All4.json
    # ----------------------------------------------------------
    """ 
    json_files = ['../FMOX-Jsons/FMOX_fall_and_tbd3d.json',
                  '../FMOX-Jsons/FMOX_fmov2.json',
                  '../FMOX-Jsons/FMOX_tbd.json']
    
    fmox_all4_json_path = '../FMOX-Jsons/FMOX_All4.json'
    combined_databases = []

    for json_file in json_files:
        try:
            with open(json_file, 'r') as file:
                data = json.load(file)
                combined_databases.extend(data.get('databases', []))  # Use .get to avoid KeyError

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file {json_file}: {e}")
            return
        except FileNotFoundError as e:
            print(f"File not found: {json_file}. Error: {e}")
            return

    combined_data = {"databases": combined_databases}  # Create a new combined structure

    with open(fmox_all4_json_path, 'w') as json_file:
        json.dump(combined_data, json_file, indent=4)

    print(f"Combined FMOX JSON saved to: {fmox_all4_json_path}")
    """

    # ------------------------- OTHER OPTIONAL CODES ------------------------------------#

    # -----------------------------------------------------------------------
    # Convert FMOv2 Run-Length Encoded (RLE) Text Files to Mask Images
    #   (Already shared as png image - could be useful if needed)
    # -----------------------------------------------------------------------
    """ The segmentation mask images for the FMOv2 dataset have been compressed into text files using the
    Run-Length Encoding (RLE) data compression technique. The following code can be used to convert these
    compressed files back into black and white segmentation images. """
    # rle_to_seg_mask_img.rle_to_mask_img()

    # ----------------------------------------------------------
    # Merge Mask Images to Visualize Trajectories
    # ----------------------------------------------------------
    """ Below code could be utilized to visualize the trajectories of the object(s) on a single image. 
    Since the FMOv2 dataset directly shares segmentation mask images, the path related to this dataset 
    is provided below as input. Each single mask image corresponding to a subsequence will be saved in 
    the "Videos/fmov2_outputs" directory as "{subfolder}_combined_segmentation_image.png". """

    # input_directory = '../Original_Dataset/FMOv2/FMOv2_gt'
    # combine_all_mask_to_single_img.combine_segmentation_images(input_directory)

    # ----------------------------------------------------------
    # TbD Dataset Bounding Box Visualization from gt.txt files
    # ----------------------------------------------------------
    # input_dir = "../Original_Dataset/TbD-3D/imgs/HighFPS_GT_depth2/"  # HighFPS_GT_depth2, fall_cube etc.
    # tbd_visualize_bboxes.tbd_vis_bbox(input_dir)


if __name__ == "__main__":
    main()

