# FMOX: Extending the ground truth labels associated with FMO datasets
=======

purpose of this repo

FMO VS NON-FMO LABELING
OBJECT SIZE LABELLING
Make easy TO USE ....

[[`Paper`](https://xxxxxxxxxxxxx)] 
[[`Notebook`](https://xxxxxxxxxxxxx)] 
[[`JSONS`](https://xxxxxxxxxxxxx)] 
[[`CSV`](https://xxxxxxxxxxxxx)] 
[[`BibTex`](#citing-imagebind)]


<!-- To appear at IMVIP 2025. For details, see the paper: 
**[FMOX : Extending the ground truth labels associated with FMO
datasets](https://facebookresearch.github.io/ImageBind/paper)**.
-->

### Object Size Category Assignment

The sizes of the objects in the public FMO datasets were calculated and "object size levels" were assigned. 
A total of five distinct level definitions were illustrated as figure below and established as follows: 
* extremely tiny ([1x1 - 8x8)), 
* tiny ([8X8,16X16)), 
* small ([16X16,32X32)),
* medium ([32x32-96x96))
* large ([96x96,&#8734;)


<img src="ExistingFMODataAnalysis/figures/proof_of_consept_obj_size.png" alt="object size image"	width="700" height="300" /> 


## Usage  

```bash
git clone https://github.com/CVMLmu/FMOX.git branch main
cd ExistingFMODataAnalysis
# for conda, create the environment using:
conda env create -n fmo_data_env -f environment.yml
conda activate fmo_data_env
```

# ----------------------------------------------------------
### Download Datasets 

The dataset is downloaded and extracted into a folder named "fmo_data".

##### Usage 
```bash 
# download_datasets_zip_files.py 
download_datasets_zip_files.download_unzip_data()
```

# ----------------------------------------------------------
### Create FMOv2 JSON Annotation File 

Contour detection is applied to the segmentation images located in "./fmo_data/FMOv2/FMOv2_gt" to obtain bounding boxes. 
The outputs of this process are saved as a video in "fmov2_outputs" folder. While obtaining the bounding boxes,
object size labels are calculated and saved in a JSON file as "fmov2_json_annotations.json" in "jsons_anns" folder. 

##### Usage 
```bash 
# create_fmov2_json.py
create_fmov2_json.get_fmov2_json()
```

# ----------------------------------------------------------
### Create JSON Annotation File For Falling Object, TbD, TbD-3D Datasets
- Acknowledgments: The code is provided (in "create_json_for_three_dataset" folder) adapted from the
[fmo-deblurring-benchmark](https://github.com/rozumden/fmo-deblurring-benchmark) and
some modifications made to fit our specific use case. For more details, please visit the repository.
- The json file called "three_fmo_data_annotations.json" saved in "json_anns" folder.
- However if you want to regenerate it, please refer below file:

##### Usage 
```bash 
# create_json_for_three_dataset.py
json_ann.create_json_for_three_dataset() 
```

# ----------------------------------------------------------
### JSON Annotation Analysis - Visualizations

Combine the 2 json annotation files created.
```bash 
    # if you create json files again combine with:
    json1 = "./json_anns/three_fmo_data_annotations.json"
    json2 = "./json_anns/fmov2_json_annotations.json"
    json_annotation_analysis.combine_and_save_jsons(json1, json2)   
```
Create [CSV](ExistingFMODataAnalysis/json_anns/json_annotation_analysis_output.csv) file from json annotation file. 


```bash 
    # Create csv file to save information e.g. "Main Dataset", "Subsequence", "Total Frame Number",
    # "FMO Exists Frame Number", "Average Object Size", "Object Size Levels" for table and graphs.
    # It will saved in "json_anns". Similar to table below ..
    json_annotation_analysis.json_to_csv()
```

#### Stacked bar chart for visualization object size levels for each dataset and subsequences.
```bash 
# csv_to_graphics.py
csv_to_graphics.visualize_object_size_levels() 
```

#### Scatter plot for comparing "Number of FMO exist frame" vs. "Number of total frame"
```bash 
# csv_to_graphics.py
csv_to_graphics.cvs_viz2() 
```

# ----------------------------------------------------------
### Access Bounding Boxes from a JSON Annotation File

##### Usage 
```bash 

# access_json_bboxes.py
access_json_bboxes.access_bboxes()
```
<img src="ExistingFMODataAnalysis/figures/json_to_bbox.png" alt="json_to_bbox" width="700" height="500" /> 

# ----------------------------------------------------------
### Merge Mask Images to Visualize Trajectories

Below code could be utilized to visualize the trajectories of the object(s) on a single image. Since the FMOv2 dataset directly shares segmentation mask images, the path related to this dataset 
is provided below as input. Each single mask image corresponding to a subsequence will be saved in the "fmov2_outputs" directory as "{subfolder}_combined_segmentation_image.png". 

##### Usage 
```bash  
#  combine_all_mask_to_single_img.py 
input_directory = './fmo_data/FMOv2/FMOv2_gt' 
combine_all_mask_to_single_img.combine_segmentation_images(input_directory)
```

# ----------------------------------------------------------
### Convert FMOv2 Run-Length Encoded (RLE) Text Files to Mask Images

The segmentation mask images for the FMOv2 dataset have been compressed into text files using the Run-Length Encoding (RLE) data compression technique. The following code can be used to convert these compressed files back into black and white segmentation images.

##### Usage 
```bash 
# rle_to_seg_mask_img.py
rle_to_seg_mask_img.rle_to_mask_img()
```

# ----------------------------------------------------------
### Metadata File

The project includes a **JSON metadata file** [here](FMOX-code/FMOX.json) that contains essential information for understanding the work context, including its attributes, original dataset papers, and other relevant details.  Please refer to the JSON metadata file for more information.

### Citing 
##### If you are using this repo in your research or applications, please cite using this BibTeX:
```bibtex
@article{xxxxxxxxxxxxxxxxxxxx,
  title={FMOX:Extending the ground truth labels associated with FMO datasets},
  author={Senem Aktas, Rozenn Dahyot, John McDonald, Charles Markham},
  conference={xxxxxxxxx},
  year={2025}
}
```

# Thank You Message for the Matas Team Regarding the Dataset ? something like below
## Acknowledgments

I would like to thank the following sources for providing the datasets used in this project:

- **[Dataset Name](URL)**: A brief description of the dataset and its significance.
- **[Another Dataset Name](URL)**: A brief description of this dataset as well.

Your contributions have been invaluable to the success of this project!

