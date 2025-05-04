import os
import json

def find_correspondence_in_json(search_json, db_name, subdb_name, image_file_name):
    for database in search_json["databases"]:
        if database.get("db_name") == db_name:
            for sub_dataset in database.get("sub_datasets", []):
                if sub_dataset.get("subdb_name") == subdb_name:
                    for image in sub_dataset.get("images", []):
                        if image.get("image_file_name") == image_file_name:
                            # Found the image, return both image_file_name and bbox
                            bbox = image.get("annotations", [{}])[0].get("bbox_xyxy", None)  # first annotation's bbox
                            return image["image_file_name"], bbox
    return None,None


fmox_json_path = "../FMOX-Jsons/FMOX_All4.json"
efficienttam_json_path = "../EfficientTAM-Jsons/efficienttam_All4.json "

# Open both JSON files simultaneously
with open(efficienttam_json_path, 'r') as file1, open(fmox_json_path, 'r') as file2:
    efficienttam_data = json.load(file1)
    fmox_data = json.load(file2)

# keep both data as [[fmox_data_bbox][efficienttam_data_bbox]] according to image base match
fmox_plus_efficienttam_bbox = []

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
            if len(sub_dataset["images"]) != 0:
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

                        # convert box formats, might be in the format [x, y, width, height].
                        # as in defined	bboxes = np.loadtxt(os.path.join(folder,'gt_bbox',seqname + '.txt'))

                        fmox_bboxes.append(bboxes)
                        efficienttam_bboxes.append(effcientbbox)

        # after this - calciou processes will be called ....
        fmox_plus_efficienttam_bbox.append(fmox_bboxes,efficienttam_bboxes)
        print("db_name", database["dataset_name"], "subdb_name", sub_dataset["subdb_name"])
        print("fmox_plus_efficienttam_bbox", fmox_plus_efficienttam_bbox)

