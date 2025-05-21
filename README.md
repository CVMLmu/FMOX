# Benchmarking EfficientTAM on FMO datasets
<!-- # FMOX: Extending the ground truth labels associated with FMO datasets -->

[[`Paper`](https://xxxxxxxxxxxxx)] 
[[`FMOX Metadata`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/FMOX.json)] 
[[`FMOX JSONS`](https://github.com/CVMLmu/FMOX/tree/main/FMOX-code/FMOX-Jsons)] 
[[`BibTex`](#citing-imagebind)]

Result of Analysis: [[`FMOX All4 Statistics`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/use-FMOX/FMOX_All4_statistics.csv)] 
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

### Installation 

```bash
git clone https://github.com/CVMLmu/FMOX.git branch main
cd FMOX
# for conda, create the environment using:
conda env create -n fmo_data_env -f environment.yml
conda activate fmo_data_env
```

### FMOX Object Size Categories

The sizes of the objects in the public FMO datasets were calculated and "object size levels" were assigned. 
A total of five distinct level defined as below: 

| Extremely Tiny        | Tiny                 | Small                | Medium               | Large               |
|----------------------|----------------------|----------------------|----------------------|---------------------|
| [1 × 1, 8 × 8)       | [8 × 8, 16 × 16)     | [16 × 16, 32 × 32)   | [32 × 32, 96 × 96)   | [96 × 96, ∞)        |

*Table: FMOX object size categories.*

### Structure of FMOX

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
              "annotations": ["bbox_xyxy": [.....], "object_wh": [.....], "size_category": "...." ]
            }
          ]
        }
      ]
    }
  ]
}
```

### Average TIoU Performance Comparison

This table compares the average TIoU $(\uparrow)$ performance of various studies on FMO datasets. Evaluation of the EfficientTAM has been done with FMOX Json. The best results are indicated with $^*$ and the second-best results with $^{**}$. 

| Datasets        | Defmo [Rozumnyi et al., 2021] | FmoDetect [Rozumnyi et al., 2021] | TbD [Kotera et al., 2019]          | TbD-3D [Rozumnyi et al., 2020]       | EfficientTAM [Xiong et al., 2024] |
|------------------|--------|------------------|------------------|------------------|----------|
| Falling Object    | 0.684** | N/A              | 0.539            | 0.539            | 0.7093*  |
| TbD               | 0.550** | (a) 0.519 (b) 0.715* | 0.542            | 0.542            | 0.4546   |
| TbD-3D           | 0.879* | N/A              | 0.598            | 0.598            | 0.8604** |

(a) Real-time with trajectories estimated by the network. (b) With the proposed deblurring. N/A indicates "Not defined".

### FMOX Metadata

The project includes a **JSON metadata file** [here](FMOX-code/FMOX.json) that contains essential information for understanding the work context, including its attributes, original dataset papers, and other relevant details.  Please refer to the JSON metadata file for more information.

### Citing 
##### If you are using this repo in your research or applications, please cite using this BibTeX:
```bibtex
@article{xxxxxxxxxxxxxxxxxxxx,
  title={Benchmarking EfficientTAM on FMO datasets},
  author={Senem Aktas, Charles Markham, John McDonald, Rozenn Dahyot},
  conference={xxxxxxxxx},
  year={2025}
}
```
####  Thank You Message for the Matas Team Regarding the Dataset ? something like below
### Acknowledgments

I would like to thank the following sources for providing the datasets used in this project:

- **[Dataset Name](URL)**: A brief description of the dataset and its significance.
- **[Another Dataset Name](URL)**: A brief description of this dataset as well.

Your contributions have been invaluable to the success of this project!

