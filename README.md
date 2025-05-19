# Benchmarking EfficientTAM on FMO datasets
<!-- # FMOX: Extending the ground truth labels associated with FMO datasets -->

[[`Paper`](https://xxxxxxxxxxxxx)] 
[[`FMOX Metadata`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/FMOX.json)] 
[[`FMOX JSONS`](https://github.com/CVMLmu/FMOX/tree/main/FMOX-code/FMOX-Jsons)] 
[[`BibTex`](#citing-imagebind)]

Analysis: [[`FMOX All4 Statistics`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/use-FMOX/FMOX_All4_statistics.csv)] 
[[`EfficientTAM TIoU`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/use-FMOX/EfficientTAM_averageTIoU.csv)] 

Notebooks : [[`Create FMOX`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/create-FMOX/create_jsons_main.ipynb)] 
[[`Use FMOX`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/use-FMOX/fmox_main.ipynb)] 


This repository is dedicated to enhancing a dataset by adding detailed labels such as object categories and bounding box
coordinates. and providing structured JSON annotations for easy integration and usage in machine learning projects. 
The JSON format allows for seamless compatibility with various machine learning frameworks, making it easier for developers
and researchers to utilize the dataset in their applications.

<!--  purpose of this repo
FMO VS NON-FMO LABELING
OBJECT SIZE LABELLING
Make easy TO USE ....

To appear at IMVIP 2025. For details, see the paper: 
**[FMOX : Extending the ground truth labels associated with FMO
datasets](https://facebookresearch.github.io/ImageBind/paper)**.
-->

### FMOX Object Size Categories

The sizes of the objects in the public FMO datasets were calculated and "object size levels" were assigned. 
A total of five distinct level defined as below: 

| Extremely Tiny        | Tiny                 | Small                | Medium               | Large               |
|----------------------|----------------------|----------------------|----------------------|---------------------|
| [1 × 1, 8 × 8)       | [8 × 8, 16 × 16)     | [16 × 16, 32 × 32)   | [32 × 32, 96 × 96)   | [96 × 96, ∞)        |

*Table: FMOX object size categories.*

### Structure of FMOX

This section describes the structure of the FMOX dataset in JSON format.

```json
{
  "databases": [
    {
      "dataset_name": "Falling_Object",
      "version": "1.0",
      "description": "Falling_Object annotations.",
      "sub_datasets": [
        {
          "subdb_name": "v_box_GTgamma",
          "images": [
            {
              "img_index": 1,
              "image_file_name": "00000027.png",
              "annotations": [
                {
                  "bbox_xyxy": [161, 259, 245, 333],
                  "object_wh": [84, 74],
                  "size_category": "medium"
                }
              ]
            },
            {
              "img_index": 2,
              "image_file_name": "00000028.png",
              "annotations": [ /* additional annotations here */ ]
            }
          ]
        }
      ]
    }
  ]
}
```

### To use FMOX refer to : xx and xxx

        

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
### Acknowledgments

I would like to thank the following sources for providing the datasets used in this project:

- **[Dataset Name](URL)**: A brief description of the dataset and its significance.
- **[Another Dataset Name](URL)**: A brief description of this dataset as well.

Your contributions have been invaluable to the success of this project!

