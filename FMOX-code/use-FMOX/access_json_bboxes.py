import os
import json
import cv2
import matplotlib.pyplot as plt

def access_bboxes(data_folder, fmox_json_path, image_limit=None):
    dataset_paths = {
        "Falling_Object": "Falling_Object/imgs/",
        "FMOv2": "FMOv2/imgs/",
        "TbD": "TbD/imgs/",
        "TbD-3D": "TbD-3D/imgs/"
    }

    with open(fmox_json_path, 'r') as json_file:
        data = json.load(json_file)

    # Iterate through the databases
    for database in data["databases"]:
        dataset_name = database["dataset_name"]

        # Reach the main dataset folder according to "dataset_name"
        main_data_path = dataset_paths.get(dataset_name, 'default_value')

        # Iterate through the sub-datasets
        for sub_dataset in database["sub_datasets"]:
            subdb_name = sub_dataset["subdb_name"]
            sub_data_path = os.path.join(data_folder, main_data_path, subdb_name)

            # Note: some sequences do not have annotations (e.g., fmov2 swaying), so skip them.
            if len(sub_dataset["images"]) != 0:
                # Create a set of image file names from the sub_dataset for quick lookup
                image_file_names = {image["image_file_name"] for image in sub_dataset["images"]}

                # Initialize a counter for displayed images
                displayed_images = 0

                # Iterate through all files in the sub_data_path
                for file_name in os.listdir(sub_data_path):
                    # Check if the file is a PNG or JPG image
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(sub_data_path, file_name)

                        img = cv2.imread(image_path)

                        # Check if the current file name matches any in the dataset
                        if file_name in image_file_names:
                            # Find the corresponding image entry in the dataset
                            for image in sub_dataset["images"]:
                                if image["image_file_name"] == file_name:
                                    # Iterate through the annotations - multiple objects could be present
                                    for annotation in image.get("annotations", []):
                                        current_bbox = annotation["bbox_xyxy"]  # format is [x1, y1, x2, y2]
                                        x1, y1, x2, y2 = map(int, current_bbox)  # Convert to integers for drawing

                                        # Draw bounding box on the image - red color with thickness 2
                                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                            # Convert BGR image to RGB
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            plt.imshow(img_rgb)
                            plt.title("Image with Bounding Boxes")
                            plt.axis('off')  # Hide axis ticks
                            plt.show()

                            # Increment the displayed images counter
                            displayed_images += 1

                            # Check if the limit has been reached
                            if image_limit is not None and displayed_images >= image_limit:
                                break  # Exit the loop if the limit is reached

                # Break the outer loop if the limit has been reached
                if image_limit is not None and displayed_images >= image_limit:
                    break

# Example usage
# data_folder = "path_to_data_folder"
# fmox_json_path = "path_to_json_file"
# access_bboxes(data_folder, fmox_json_path, image_limit=5)  # Display up to 5 images
