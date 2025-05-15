import cv2
import numpy as np
import os


def combine_segmentation_images(input_directory):
    # Iterate through each subfolder in the main directory
    for subfolder in os.listdir(input_directory):
        subfolder_path = os.path.join(input_directory, subfolder)

        # Check if the path is a directory
        if os.path.isdir(subfolder_path):
            # Get a list of all image files in the subfolder
            image_files = [f for f in os.listdir(subfolder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

            # Initialize a list to hold the images
            images = []

            # Read each image and append it to the list
            for image_file in image_files:
                image_path = os.path.join(subfolder_path, image_file)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
                if img is not None:
                    images.append(img)
                else:
                    print(f"Warning: {image_file} could not be read.")

            # Check if any images were read
            if not images:
                print(f"No images found in the subfolder: {subfolder}")
                continue

            # Combine images into one
            combined_image = np.zeros_like(images[0])  # Initialize a blank image with the same shape as the first image

            for img in images:
                combined_image = cv2.bitwise_or(combined_image, img)  # Combine using bitwise OR

            # Save the combined image in the main directory with the subfolder name
            output_dir = "../Videos/fmov2_outputs/"
            output_image_path = os.path.join(output_dir, f"{subfolder}_combined_segmentation_image.png")
            cv2.imwrite(output_image_path, combined_image)
            print(f"Combined image saved to {output_image_path}")


# # Example usage
# input_directory = 'FMOv2_gt'  # Replace with your main directory
# combine_segmentation_images(input_directory)
