import os
import cv2
import json
import numpy as np

class JsonFMO:

    def __init__(self, each_inner_folder_name, images_path, db_owner_name, output_folder):
        self.each_inner_folder_name = each_inner_folder_name
        self.images_path = images_path
        self.db_owner_name = db_owner_name
        self.output_folder = output_folder
        self.fps = 5

        # (0,0) because object could be a point as well. So it wont have w or height ....
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

    @staticmethod
    def is_contour_inside_area(contour, area):
        """  Check if the given contour is completely inside the specified area.
        Parameters: contour (numpy.ndarray): The contour points (Nx2 array).
                    area (tuple): The area defined by (x_min, y_min, x_max, y_max).
        Returns:  bool: True if the contour is inside the area, False otherwise. """
        x_min, y_min, x_max, y_max = area

        # Check each point in the contour
        for point in contour:
            x, y = point
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                return False  # Point is outside the area

        return True  # All points are inside the area

    def get_sub_dataset_entry(self):
        self.images_path = self.images_path
        out_vid_name = os.path.basename(os.path.normpath(self.images_path))
        # video_name = self.output_folder + self.db_owner_name + "_" + str(out_vid_name) + "_fmov2_out.avi"
        video_name = self.output_folder + str(out_vid_name) + "_fmov2_out.avi"

        images = [img for img in os.listdir(self.images_path)
                  if img.endswith(".jpg")  or img.endswith(".jpeg") or img.endswith("png")]

        frame = cv2.imread(os.path.join(self.images_path, images[0]))
        img_height, img_width, layers = frame.shape
        video = cv2.VideoWriter(video_name, 0, self.fps, (img_width, img_height))

        #  Loop to fill in the sub-datasets
        sub_dataset_entry = {
            "subdb_name": self.each_inner_folder_name,
            "total_frame_num": len(images),
            "images": []
        }

        # Appending the images to the video one by one
        for img_index, image_name in enumerate(images):
            image = cv2.imread(os.path.join(self.images_path, image_name))
            annotations = []

            # ------------------------ specific processes ----------------------------------------
            # extra mask from different object only frame in 00000010.png - do not count it
            if  self.each_inner_folder_name == "ping_pong_paint" and image_name == "00000010.png":
                continue
            # ------------------------------------------------------------------------------------

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            # Apply a binary threshold to get a binary image - adjust the threshold value (e.g., 200)
            _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
            cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # Green contours

            # Get the bounding rectangle of the contour
            for each_contour in contours:
                x, y, w, h = cv2.boundingRect(each_contour)
                y_min = int(y)
                x_min = int(x)
                x_max = int(x+w)
                y_max = int(y+h)
                cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

                obj_width = int(x_max) - int(x_min)
                obj_height = int(y_max) - int(y_min)

                # ------------------------ specific processes ----------------------------------------
                # extra masks out of target object - do not count masks from defined area
                if self.each_inner_folder_name == "william_tell":
                    current_contour = np.array([[int(x_min), int(y_min)],
                                        [int(x_max), int(y_min)],
                                        [int(x_max), int(y_max)],
                                        [int(x_min), int(y_max)]], dtype=np.int32)

                    do_not_count_area = (1105, 318, 1611, 1028)  # (x_min, y_min, x_max, y_max)

                    result = self.is_contour_inside_area(current_contour, do_not_count_area)
                    if result is True:   # True if the contour is inside the area
                        # print("Pass if the contour inside the area? ", result)
                        continue

                # ------------------------------------------------------------------------------------

                annotations.append({
                    "bbox_xyxy": [int(x_min), int(y_min), int(x_max), int(y_max)],
                    "object_wh": (obj_width, obj_height),
                    "size_category": self.get_obj_size_category2(obj_width, obj_height)
                })

            if len(contours) != 0:   # if there is a value then save it...
                # Create the image entry
                image_entry = {
                    "img_index": img_index,
                    "image_file_name": image_name,
                    "annotations": annotations
                }

                # Add the image entry to the sub-dataset
                sub_dataset_entry["images"].append(image_entry)

            video.write(image)
        cv2.destroyAllWindows()
        video.release()
        # print("done...")

        return sub_dataset_entry


def get_fmov2_json():
    db_owner = "fmov2"
    whole_images_folder = "../Original_Dataset/FMOv2/FMOv2_gt"
    out_folder = "../Videos/fmov2_outputs/contour_videos/"
    os.makedirs(out_folder, exist_ok=True)

    # Initialize the main data structure
    data = {
        "databases": []
    }
    dataset_name = "FMOv2"

    db_entry = {
        "dataset_name": dataset_name,
        "version": "1.0",
        "description": f"{dataset_name} containing bounding box and object size annotations.",
        "sub_datasets": []
    }

    for each_inner_folder_name in os.listdir(whole_images_folder):
        each_inner_folder_path = str(whole_images_folder) + "/" + str(each_inner_folder_name)
        sub_dataset_entry = JsonFMO(each_inner_folder_name,each_inner_folder_path, db_owner, out_folder).get_sub_dataset_entry()

        # Add the sub-dataset entry to the database
        db_entry["sub_datasets"].append(sub_dataset_entry)

    # Add the database entry to the main data structure
    data["databases"].append(db_entry)

    save_path = "../FMOX-Jsons/FMOX_fmov2.json"   # Save the data to a JSON file
    with open(str(save_path), 'w') as json_file:
        json.dump(data, json_file, indent=4)

    # print("Dataset path: {}, JSON saved in: {}, Videos saved in: {}".format(whole_images_folder, save_path, out_folder))
    print("JSON saved in: {}, Videos saved in: {}".format(save_path, out_folder))

# get_fmov2_json()
