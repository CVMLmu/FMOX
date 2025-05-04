import os
import json


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

fmox_json_path = "../FMOX-Jsons/FMOX_All4.json"
efficienttam_json_path = "../EfficientTAM-Jsons/efficienttam_All4.json"

# Open both JSON files simultaneously
with open(efficienttam_json_path, 'r') as file1, open(fmox_json_path, 'r') as file2:
    efficienttam_data = json.load(file1)
    fmox_data = json.load(file2)

# keep both data as [[fmox_data_bboxes][efficienttam_data_bboxes]] according to image base match
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
            # note: some sequences does not have annotations (e.g. fmov2 swaying) so skip them.
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
                        bboxes = convert_bbox_xyxy_to_xywh(bboxes)
                        effcientbbox = convert_bbox_xyxy_to_xywh(effcientbbox)

                        fmox_bboxes.append(bboxes)
                        efficienttam_bboxes.append(effcientbbox)

        # after this - calciou processes will be called ....
        if len(efficienttam_bboxes) and len(fmox_bboxes) != 0:
            fmox_plus_efficienttam_bbox.append([fmox_bboxes, efficienttam_bboxes])
            print("db_name", database["dataset_name"], "subdb_name", sub_dataset["subdb_name"])
            print("len(efficienttam_bboxes)", len(efficienttam_bboxes), "len(fmox_bboxes)", len(fmox_bboxes),
                  '\n\n')

            import calciou
            calciou.evaluate_on(sub_dataset["subdb_name"], fmox_bboxes, efficienttam_bboxes)


