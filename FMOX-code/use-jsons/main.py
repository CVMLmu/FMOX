import os
import download_datasets_zip_files
import combine_all_mask_to_single_img
import create_fmov2_json
import rle_to_seg_mask_img
import create_json_for_three_dataset.create_json_via_benchmark_loader as json_ann
import json_annotation_analysis
import csv_to_graphics
import access_json_bboxes


def main():

    # ---------------------------------
    # Download Datasets
    # ---------------------------------
    """ The dataset is downloaded and extracted into a folder named "fmo_data". """

    # download_datasets_zip_files.download_unzip_data()

    # ----------------------------------------------------------
    # Create FMOv2 JSON Annotation File
    # ----------------------------------------------------------
    """ Contour detection is applied to the segmentation images located in "./fmo_data/FMOv2/FMOv2_gt"
     to obtain bounding boxes. The outputs of this process are saved as a video in "fmov2_outputs" folder. 
     While obtaining the bounding boxes, object size labels are calculated and saved in a JSON file as 
     "fmov2_json_annotations.json" in "jsons_anns" folder. """

    # create_fmov2_json.get_fmov2_json()

    # ---------------------------------------------------------------------
    # Create JSON Annotation File For Falling Object, TbD, TbD-3D Datasets
    # ---------------------------------------------------------------------
    """ - Acknowledgments: The code is provided (in "create_json_for_three_dataset" folder) adapted 
    from the [fmo-deblurring-benchmark](https://github.com/rozumden/fmo-deblurring-benchmark) and
    some modifications made to fit our specific use case. For more details, please visit the repository.
    - The json file called "three_fmo_data_annotations.json" saved in "json_anns" folder.
    - However if you want to regenerate it, please refer below file:"""

    # json_ann.create_json_for_three_dataset()

    # ----------------------------------------------------------
    # JSON Annotation Analysis - Visualizations
    # ----------------------------------------------------------
    # Combine the 2 json annotation files created.
    # json1 = "./json_anns/three_fmo_data_annotations.json"
    # json2 = "./json_anns/fmov2_json_annotations.json"
    # json_annotation_analysis.combine_and_save_jsons(json1, json2)

    # Create csv file to save information e.g. "Main Dataset", "Subsequence", "Total Frame Number",
    # "FMO Exists Frame Number", "Average Object Size", "Object Size Levels" for table and graphs.
    # It will saved in "json_anns". TODO: average size calculation is proper to calculate?
    # json_annotation_analysis.json_to_csv()

    # Different kind of visualization functions : Bar plot, Scatter plot, Box plot etc.
    #  Scatter plot for comparing "Number of FMO exist frame" vs. "Number of total frame"
    # csv_to_graphics.cvs_viz2()
    # Stacked bar chart for visualization object size levels for each dataset and subsequences.
    # csv_to_graphics.visualize_object_size_levels()

    # ----------------------------------------------------------
    # Access Bounding Boxes from a JSON Annotation File
    # ----------------------------------------------------------
    # access_json_bboxes.access_bboxes()

    # ----------------------------------------------------------
    # Merge Mask Images to Visualize Trajectories
    # ----------------------------------------------------------
    """ Below code could be utilized to visualize the trajectories of the object(s) on a single image. 
    Since the FMOv2 dataset directly shares segmentation mask images, the path related to this dataset 
    is provided below as input. Each single mask image corresponding to a subsequence will be saved in 
    the "fmov2_outputs" directory as "{subfolder}_combined_segmentation_image.png". """

    # input_directory = './fmo_data/FMOv2/FMOv2_gt'
    # combine_all_mask_to_single_img.combine_segmentation_images(input_directory)

    # -----------------------------------------------------------------------
    # Convert FMOv2 Run-Length Encoded (RLE) Text Files to Mask Images
    #   (Already shared as png image - could be useful if needed)
    # -----------------------------------------------------------------------
    """ The segmentation mask images for the FMOv2 dataset have been compressed into text files using the
    Run-Length Encoding (RLE) data compression technique. The following code can be used to convert these
    compressed files back into black and white segmentation images. """
    # TODO: in their segmentation masks there are gray masks as well - is it important?
    # rle_to_seg_mask_img.rle_to_mask_img()


if __name__ == "__main__":
    main()

