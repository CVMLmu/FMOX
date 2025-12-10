# Introduction

In this repo, Fast Moving Object (FMO) datasets (`FMOv2`, `TbD-3D`, `TbD` and `Falling Objects`, all available at [https://cmp.felk.cvut.cz/fmo/](https://cmp.felk.cvut.cz/fmo/)) with additional ground truth information 
in JSON format (our new metadata is called FMOX) are provided and used for benchmarking trackers.

:newspaper: **If you are using this repo in your research or applications, please cite our papers related to this work (published in IMVIP 2025 and AICS 2025).** 


# Benchmarking SAM2 based trackers on FMOX (AICS 2025)

[[`Paper`](FMOXaics2025.pdf)] 

```bibtex
@inproceedings{AICS2025-Aktas,
title={Benchmarking SAM2-based Trackers on FMOX},
author={Senem Aktas and Charles Markham and John McDonald and Rozenn Dahyot},
booktitle={33rd International Conference on Artificial Intelligence and Cognitive Science (AICS 2025)},
address={Dublin, Ireland},
year={2025},
month={December},
pages={},
doi={},
url={web=https://aicsconf.org/},
abstract={},
keywords={},
note={},
}
```

# Benchmarking EfficientTAM on FMO datasets (IMVIP 2025)

[[`Paper`](FMOXimvip2025.pdf)] 
[[`Code`](https://github.com/CVMLmu/FMOX/)] [[`Arxiv`](https://arxiv.org/abs/2509.06536)]

 
In this repo, we extend Fast Moving Object (FMO) datasets (`FMOv2`, `TbD-3D`, `TbD` and `Falling Objects`, all available at [https://cmp.felk.cvut.cz/fmo/](https://cmp.felk.cvut.cz/fmo/)) with additional ground truth information 
in JSON format (our new metadata is called FMOX). 
The provided FMOX JSON format allows for seamless compatibility with various machine learning frameworks, making it easier for developers
and researchers to utilize the dataset in their applications.
With FMOX, we test a recently proposed foundational model for tracking ([EfficientTAM](https://yformer.github.io/efficient-track-anything/))  showing that its performance compares well with the [pipelines originally  developed for these FMO datasets](https://cmp.felk.cvut.cz/fmo/).

Scripts provided in this repo allow to download all FMO datasets,  create `json` metadata, and assess object tracking with EfficientTAM  using with TIoU metric.  



```bibtex
@inproceedings{FMOX_AKTAS2025,
  title={Benchmarking EfficientTAM on FMO datasets},
  author={Senem Aktas and Charles Markham and John McDonald and Rozenn Dahyot},
  booktitle={Irish Machine Vision and Image Processing},
  doi={10.48550/arXiv.2509.06536},
  url={https://cvmlmu.github.io/FMOX/},
  month={September},
  pages={59-66},
  address={Ulster University, Derry-Londonderry, Northern Ireland},
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

The following notebooks can be run in that environment:
 - [`create_jsons_main.ipynb`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/create-FMOX/create_jsons_main.ipynb)  or [`main.py`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/create-FMOX/main.py) for creating FMOX JSON files in folder `FMOX-Jsons` (these are already provided in this Github).
 - [`fmox_main.ipynb`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/use-FMOX/fmox_main.ipynb) or [`fmox_main.py`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/use-FMOX/fmox_main.py) to evaluate efficientTAM  on FMOX with TIOU (i.e. all 4 FMO datasets using our JSON metadata files).  EfficientTAM results are already computed and are provided in JSON in the folder [EfficientTAM-Jsons](https://github.com/CVMLmu/FMOX/tree/main/FMOX-code/EfficientTAM-Jsons): we have used  the pretrained model `EfficientTAM-S`  with its default parameters.
 - Note: Downloading the dataset (except MATLAB files) may take some time, as all downloads are approximately 24 GB in size. If the dataset already exists in the specified location, the code will automatically skip this step.

   
### Repo tree structure

```
  environment.yml
│   LICENSE
│   README.md
│
└───FMOX-code
    │   download_datasets.py
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
                (...)
```
- [`FMOX-Jsons`](https://github.com/CVMLmu/FMOX/tree/main/FMOX-code/FMOX-Jsons)  folder contains all  metadata files contains for the FMO datasets (created with `create_jsons_main.ipynb`):
   - [`FMOX_All4.json`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/FMOX-Jsons/FMOX_All4.json): All `Falling Object`, `FMOV2`, `TbD` and `TbD-3D` dataset annotations are in that JSON file.
   - [`FMOX_fall_and_tbd3d.json`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/FMOX-Jsons/FMOX_fall_and_tbd3d.json):  `Falling Object` and `TbD-3D` dataset annotations are in that JSON file.
   - [`FMOX_fmov2.json`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/FMOX-Jsons/FMOX_fmov2.json) : Only `FMOV2` dataset annotations in a JSON file due to annotation format differences
   -  [`FMOX_tbd.json`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/FMOX-Jsons/FMOX_tbd.json): `TbD` annotations in a JSON file, annotations obtained with `fall_and_tbd3d` datasets.
   -  [`FMOX_tbd_whole_sequence.json`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/FMOX-Jsons/FMOX_tbd_whole_sequence.json) : Only `TbD` dataset annotations in a JSON file due to annotation format differences.
  
   The JSON files include the ground truth bounding boxes for the four datasets examined. Note that there are two ways to create the `TbD` JSON. The first method involves creating `FMOX_tbd.json`, which is obtained using the same code ([`create-FMOX/dataset_loader`](https://github.com/CVMLmu/FMOX/tree/main/FMOX-code/create-FMOX/dataset_loader)) as `fall_and_tbd3d.json`.  The second method involves creating `FMOX_tbd_whole_sequence.json`, which  contains bounding boxes for the entire sequence, not just for the FMOs and obtained directly from the provided original ground truth text files.
             
- [`EfficientTAM-Jsons`](https://github.com/CVMLmu/FMOX/tree/main/FMOX-code/EfficientTAM-Jsons): This folder contains the evaluation results of EfficientTAM saved in FMOX JSON format. After installing the EfficientTAM repository, the first bounding box coordinates from the FMOX JSON files for each sequence are provided to initialize the EfficientTAM tracker.   


## Additional Information

The following results  are shared in this repo (created with `fmox_main.ipynb`):
 - [`FMOX_All4_statistics.csv`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/use-FMOX/FMOX_All4_statistics.csv): This describes all the sequences from  the 4 FMO datasets. 
 - [`EfficientTAM_averageTIoU.csv`](https://github.com/CVMLmu/FMOX/blob/main/FMOX-code/use-FMOX/EfficientTAM_averageTIoU.csv): This provides the TIoU obtained with [EfficientTAM](https://yformer.github.io/efficient-track-anything/).  

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


## Acknowledgments

This Github repo was created and developped  by [Senem Aktas](https://orcid.org/0000-0002-3996-2771), and in addition it was tested by [Rozenn Dahyot](https://roznn.github.io/).  
This research was supported by funding through the [Maynooth University Hume Doctoral Awards](https://www.maynoothuniversity.ie/graduate-research-academy/john-pat-hume-doctoral-awards).  
We  would like to thank [the authors of the FMO datasets](https://cmp.felk.cvut.cz/fmo/)  for making their datasets available.  

## License:

This code is available under the [MIT](https://choosealicense.com/licenses/mit/) License.

