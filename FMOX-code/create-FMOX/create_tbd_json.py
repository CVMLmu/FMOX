import os
import cv2
import numpy as np
import json

def read_ground_truth(gt_file):
    with open(gt_file, 'r') as f:
        lines = f.readlines()

    gt_data = []
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            # x_coords = list(map(int, lines[i].strip().split()))
            x_coords = list(map(lambda x: int(float(x)), lines[i].strip().split()))
            # y_coords = list(map(int, lines[i + 1].strip().split()))
            y_coords = list(map(lambda x: int(float(x)), lines[i + 1].strip().split()))

            # Each bounding box consists of 4 points (x,y coordinates)
            bbox = []
            for j in range(len(x_coords)):
                if j < len(y_coords):  # Ensure we have both x and y
                    bbox.append((x_coords[j], y_coords[j]))
            gt_data.append(bbox)
    return gt_data


def draw_bounding_box(img, bbox, color=(0, 255, 0), thickness=2):
    # Draw the original points
    for point in bbox:
        cv2.circle(img, point, 5, (0, 0, 255), -1)  # Red circles for the points

    # Find min and max coordinates to create a rectangle
    x_coords = [p[0] for p in bbox]
    y_coords = [p[1] for p in bbox]

    # Filter out any points at (0,0) which might be padding
    valid_points = [
        (x, y) for x, y in zip(x_coords, y_coords) if not (x == 0 and y == 0)
    ]

    if valid_points:
        x_coords = [p[0] for p in valid_points]
        y_coords = [p[1] for p in valid_points]

        x_min, y_min = min(x_coords), min(y_coords)
        x_max, y_max = max(x_coords), max(y_coords)

        # Draw rectangle
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

        # Also connect the original points with lines
        pts = np.array(valid_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (255, 0, 0), 1)  # Blue polyline
    else:
        print("Warning: No valid points found to draw a bounding box")

    return img


class JsonTbD:

    def __init__(self, each_inner_folder_name, images_path, db_owner_name):
        self.each_inner_folder_name = each_inner_folder_name
        self.images_path = images_path
        self.db_owner_name = db_owner_name
        self.object_size_labels = {"extremely_tiny": ((0, 0), (8, 8)),
                                   "tiny": ((8, 8), (16, 16)),
                                   "small": ((16, 16), (32, 32)),
                                   "medium": ((32, 32), (96, 96)),
                                   "large": ((96, 96), (100000, 100000))}

    # according to object area - e.g. the area of a 32 x 32 pixel object is 1024 square pixels.
    def get_obj_size_category2(self, obj_width, obj_height):
        obj_size_category = ""
        for label_name, label_value in self.object_size_labels.items():
            start_point, end_point = label_value
            w1, h1 = start_point
            w2, h2 = end_point

            if w1 * h1 <= obj_width * obj_height < w2 * h2:
                obj_size_category = str(label_name)
            else:
                continue
        return obj_size_category

    def return_sub_dataset_entry(self):
        self.images_path = self.images_path
        images = [img for img in os.listdir(self.images_path)
                  if img.endswith(".jpg")  or img.endswith(".jpeg") or img.endswith("png")]

        #  Loop to fill in the sub-datasets
        sub_dataset_entry = {
            "subdb_name": self.each_inner_folder_name,
            "total_frame_num": len(images),
            "images": []
        }

        # -----------------------------------------
        gt_file = os.path.join(self.images_path, 'gt.txt')

        # Read ground truth data
        gt_data = read_ground_truth(gt_file)
        # ------------------------------------------

        # Appending the images to the video one by one
        for img_index, image_name in enumerate(images):
            annotations = []
            current_image_gt = gt_data[img_index]

            # Find min and max coordinates to create a rectangle
            x_coords = [p[0] for p in current_image_gt]
            y_coords = [p[1] for p in current_image_gt]

            # Filter out any points at (0,0) which might be padding
            valid_points = [
                (x, y) for x, y in zip(x_coords, y_coords) if not (x == 0 and y == 0)
            ]

            if valid_points:
                x_coords = [p[0] for p in valid_points]
                y_coords = [p[1] for p in valid_points]

                x_min, y_min = int(min(x_coords)), int(min(y_coords))
                x_max, y_max = int(max(x_coords)), int(max(y_coords))

                obj_width = int(x_max) - int(x_min)
                obj_height = int(y_max) - int(y_min)

                annotations.append({
                    "bbox_xyxy": [int(x_min), int(y_min), int(x_max), int(y_max)],
                    "object_wh": (obj_width, obj_height),
                    "size_category": self.get_obj_size_category2(obj_width, obj_height)
                })

            if len(annotations) != 0:   # if there is a value then save it...
                # Create the image entry
                image_entry = {
                    "img_index": img_index,
                    "image_file_name": image_name,
                    "annotations": annotations
                }

                # Add the image entry to the sub-dataset
                sub_dataset_entry["images"].append(image_entry)
        return sub_dataset_entry


def get_tbd_json(whole_images_folder, json_save_path):
    # whole_images_folder = "../Original_Dataset/TbD/imgs/"
    db_owner = "tbd"

    # Initialize the main data structure
    data = {
        "databases": []
    }
    dataset_name = "TbD"

    db_entry = {
        "dataset_name": dataset_name,
        "version": "1.0",
        "description": f"{dataset_name} containing bounding box and object size annotations.",
        "sub_datasets": []
    }

    for each_inner_folder_name in os.listdir(whole_images_folder):
        each_inner_folder_path = str(whole_images_folder) + "/" + str(each_inner_folder_name)
        sub_dataset_entry = JsonTbD(each_inner_folder_name,each_inner_folder_path, db_owner).return_sub_dataset_entry()

        # Add the sub-dataset entry to the database
        db_entry["sub_datasets"].append(sub_dataset_entry)

    # Add the database entry to the main data structure
    data["databases"].append(db_entry)

    # json_save_path = "../FMOX-Jsons/FMOX_tbd.json"   # Save the data to a JSON file
    with open(str(json_save_path), 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print("Dataset path: {}, JSON saved in: {}".format(whole_images_folder, json_save_path))

# if __name__ == '__main__':
#     get_tbd_json()
