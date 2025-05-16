import os
import json
import cv2
import pandas as pd


def find_correspondence_in_json(search_json, db_name, subdb_name, image_file_name):
    for database in search_json.get("databases", []):
        if database.get("dataset_name") == db_name:
            for sub_dataset in database.get("sub_datasets", []):
                if sub_dataset.get("subdb_name") == subdb_name:
                    for image in sub_dataset.get("images", []):
                        if image.get("image_file_name") == image_file_name:
                            bbox = image.get("annotations", [{}])[0].get("bbox_xyxy", None)
                            return image["image_file_name"], bbox
    return None, None

def convert_bbox_xyxy_to_xywh(bbox_xyxy):
    # Convert bbox from [x_min, y_min, x_max, y_max] to [x_min, y_min, width, height].
    if bbox_xyxy is None or len(bbox_xyxy) != 4:
        return None
    x_min, y_min, x_max, y_max = bbox_xyxy
    width = x_max - x_min
    height = y_max - y_min
    return [x_min, y_min, width, height]


def evaluate_efficienttam(dataset_path, fmox_json_path, efficienttam_json_path, averageTIoU_path):

    # Open both JSON files simultaneously
    with open(efficienttam_json_path, 'r') as file1, open(fmox_json_path, 'r') as file2:
        efficienttam_data = json.load(file1)
        fmox_data = json.load(file2)

    # keep both data as [[fmox_data_bboxes][efficienttam_data_bboxes]] according to image base match
    fmox_plus_efficienttam_bbox = []

    # save the average TIoU for each sequence - each dataset wil have a csv file
    column_names = ["Main Dataset", "Subsequence", "Sequence Average TIoU"]
    df = pd.DataFrame(columns=column_names)

    # Iterate through the databases - on fmox_data
    for database in fmox_data["databases"]:
        # Iterate through the sub-datasets
        for sub_dataset in database["sub_datasets"]:

            fmox_bboxes = []
            efficienttam_bboxes = []

            # one object initialization in effcicienttam - it is possible to initilaze others as well
            # but no ID in gt to compare-calculate IoU ...
            if sub_dataset["subdb_name"] == "more_balls":
                continue  # Skip this iteration
            else:
                # note: some sequences does not have annotations (e.g. fmov2 swaying) so skip them.
                if len(sub_dataset["images"]) != 0:
                    first_img_name = sub_dataset["images"][0]["image_file_name"]
                    start_ind = int(first_img_name.split('.')[0])
                    for image in sub_dataset["images"]:

                        # ==============================================================================================
                        db_name = database["dataset_name"]
                        subdb_name = sub_dataset["subdb_name"]
                        image_file_name = image["image_file_name"]
                        img_name, effcientbbox = find_correspondence_in_json(efficienttam_data, db_name,
                                                                             subdb_name, image_file_name)

                        # if efficienttam_data does not have bbox for corresponding frame bbox value will be 1,1,1,1 ?
                        # there will not be any intersection' just to keep index ....
                        effcientbbox = effcientbbox if effcientbbox is not None else [1, 1, 1, 1]
                        # ==============================================================================================

                        for annotation in image["annotations"]:
                            bboxes = annotation["bbox_xyxy"]

                            bboxes = convert_bbox_xyxy_to_xywh(bboxes)
                            effcientbbox = convert_bbox_xyxy_to_xywh(effcientbbox)

                            fmox_bboxes.append(bboxes)
                            efficienttam_bboxes.append(effcientbbox)

            # call TIoU ......
            if len(efficienttam_bboxes) and len(fmox_bboxes) != 0:
                fmox_plus_efficienttam_bbox.append([fmox_bboxes, efficienttam_bboxes])
                # print("\nDataset Name", database["dataset_name"], "Subsequence Name", sub_dataset["subdb_name"])

                subseq_folder = dataset_path + database["dataset_name"] + "/imgs/" + sub_dataset["subdb_name"] + "/"
                all_files = os.listdir(subseq_folder)
                image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
                image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
                original_first_Img = None
                if image_files:
                    first_image = image_files[0]  # Get the first image file name
                    first_image_path = os.path.join(subseq_folder, first_image)  # Full path to the first image

                    original_first_Img = cv2.imread(first_image_path)

                import calciou
                df_updated = calciou.evaluate_on(df, database["dataset_name"], sub_dataset["subdb_name"],
                                                 original_first_Img, fmox_bboxes, efficienttam_bboxes, start_ind)
                df = df_updated

    # print("Average TIoU Samples: ", df.head())
    # Save the DataFrame to a CSV file - Set index=False to avoid saving the index as a column
    df.to_csv(averageTIoU_path, index=False)

    efficientTAM_traj_vis_path = "./efficientTAM_traj_vis/"
    print("EfficientTAM trajectory Estimations Saved in: ", efficientTAM_traj_vis_path)
    print("EfficientTAM TIOU Saved in: ", averageTIoU_path)

