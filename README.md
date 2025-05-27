# Benchmarking EfficientTAM on FMO datasets

[[`Paper`](https://xxxxxxxxxxxxx)] 
[[`Code`](https://github.com/CVMLmu/FMOX/)] 
[[`FMOX JSON Metadata`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/FMOX.json)] 
 



In this repo, we extend Fast Moving Object (FMO) datasets (available at [https://cmp.felk.cvut.cz/fmo/](https://cmp.felk.cvut.cz/fmo/)) with additional ground truth information 
in JSON format (our new metadata is called FMOX). 
The provided FMOX JSON format allows for seamless compatibility with various machine learning frameworks, making it easier for developers
and researchers to utilize the dataset in their applications.
With FMOX, we test a recently proposed foundational model for tracking ([EfficientTAM](https://yformer.github.io/efficient-track-anything/))  showing that its performance compares well with the [pipelines originally  developed for these FMO datasets](https://cmp.felk.cvut.cz/fmo/) .

**If you are using this repo in your research or applications, please cite our paper related to this work:** 

```bibtex
@techreport{FMOX_AKTAS2025,
  title={Benchmarking EfficientTAM on FMO datasets},
  author={Senem Aktas and Charles Markham and John McDonald and Rozenn Dahyot},
  doi={upcoming},
  url={upcoming},
  month={June},
  institution={Maynooth University Ireland},
  year={2025},
}
```

## Installation 

### Getting started
```bash
git clone https://github.com/CVMLmu/FMOX.git branch main
cd FMOX
# for conda, create the environment using:
conda env create -n fmo_data_env -f environment.yml
conda activate fmo_data_env
```

### Notebooks

The following **Notebooks** can be run in that environment:
 - [`create_jsons_main.ipynb`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/create-FMOX/create_jsons_main.ipynb) for creating FMOX: this is slow as all datasets are downloaded (~24GB)
 - [`fmox_main.ipynb`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/use-FMOX/fmox_main.ipynb) to test efficientTAM  on FMOX


   
### Repo tree structure

```
  environment.yml
│   LICENSE
│   README.md
│
└───FMOX-code
    │   download_datasets.py
    │   FMOX.json
    │   __init__.py
    │
    ├───create-FMOX
    │   │   combine_all_mask_to_single_img.py
    │   │   create_fmov2_json.py
    │   │   create_jsons_main.ipynb
    │   │   create_tbd_json.py
    │   │   main.py
    │   │   rle_to_seg_mask_img.py
    │   │   tbd_visualize_bboxes.py
    │   │
    │   └───dataset_loader
    │           create_json_via_benchmark_loader.py
    │           loaders_helpers.py
    │           reporters.py
    │
    ├───EfficientTAM-Jsons
    │       efficienttam_All4.json
    │       efficienttam_falling.json
    │       efficientTam_fmov2.json
    │       efficienttam_tbd3d.json
    │       efficienttam_tdb.json
    │
    ├───FMOX-Jsons
    │       FMOX_All4.json
    │       FMOX_fall_and_tbd3d.json
    │       FMOX_fmov2.json
    │       FMOX_tbd.json
    │       FMOX_tbd_whole_sequence.json
    │
    └───use-FMOX
        │   access_json_bboxes.py
        │   calciou.py
        │   csv_to_graphics.py
        │   efficientam_evaluation.py
        │   EfficientTAM_averageTIoU.csv
        │   FMOX_all4_json_to_CSV.py
        │   FMOX_All4_statistics.csv
        │   fmox_main.ipynb
        │   fmox_main.py
        │   size_label_bar.png
        │   size_label_count.py
        │   vis_trajectory.py
        │   __init__.py
        │
        └───efficientTAM_traj_vis
                efficientTAM_traj_Falling_Object_v_box_GTgamma.jpg
                ...
```

## Result files

The following results  are shared in this repo (created with `fmox_main.ipynb`):
 - [`FMOX_All4_statistics.csv`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/use-FMOX/FMOX_All4_statistics.csv): This describes all the sequences from  the 4 FMO datasets. 
 - [`EfficientTAM_averageTIoU.csv`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/use-FMOX/EfficientTAM_averageTIoU.csv): This provides the TIoU obtained with [EfficientTAM](https://yformer.github.io/efficient-track-anything/).  
 - [FMOX.json](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/FMOX.json) provides the metadata in `json` format for all 4 datasets. In addition,  an individual `json` is provided for each FMO dataset in folder  [`FMOX-Jsons`](https://github.com/CVMLmu/FMOX/tree/main/FMOX-code/FMOX-Jsons)

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



## License:

This code is available under the [MIT](https://choosealicense.com/licenses/mit/) License.

