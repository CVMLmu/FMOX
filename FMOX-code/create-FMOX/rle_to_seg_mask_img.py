import os
import cv2
import numpy as np


def read_ground_truth(file_path):
    """Read the ground truth file and return the header and frame data."""
    with open(file_path, 'r', encoding='ascii') as file:
        lines = file.readlines()

    # Normalize line endings to Unix-style
    lines = [line.replace('\r\n', '\n').replace('\r', '\n') for line in lines]

    # Read the header
    header = list(map(int, lines[0].strip().split()))
    W, H, F, O, L = header  # Unpack header values
    frame_data = [list(map(int, line.strip().split())) for line in lines[1:L + 1]]

    return W, H, F, O, L, frame_data

def fill_image_with_runs(image, run_lengths):
    """Fill the image based on the run-length encoding."""
    pixel_index = 0
    total_pixels = image.size
    current_color = 0  # 0 for black, 255 for white

    for length in run_lengths:
        for _ in range(length):
            if pixel_index < total_pixels:  # Ensure we don't go out of bounds
                image[pixel_index // image.shape[1], pixel_index % image.shape[1]] = current_color
                pixel_index += 1
        current_color = 255 if current_color == 0 else 0  # Alternate color

    # Fill the remaining pixels with the last color if needed
    if pixel_index < total_pixels:
        image[pixel_index // image.shape[1], pixel_index % image.shape[1]] = current_color

def create_combined_image(W, H, frame_data):
    """Create a single image that combines all annotations."""
    combined_image = np.zeros((H, W), dtype=np.uint8)  # Initialize combined image

    for idx, frame in enumerate(frame_data):
        run_lengths = frame[2:]  # The lengths of the runs
        image = np.zeros((H, W), dtype=np.uint8)  # Create an empty image for the current frame
        fill_image_with_runs(image, run_lengths)

        # Overlay the current frame onto the combined image
        combined_image = cv2.bitwise_or(combined_image, image)

    return combined_image

def rle_to_mask_img(input_folder,output_folder):

    # input_folder = "../Original_Dataset/FMOv2/"
    # output_folder = "../Videos/fmov2_outputs/rleTXT_to_video/"

    # Iterate through the files in the specified directory
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            base_name, extension = os.path.splitext(filename)
            input_rle_txt_file = input_folder + filename
            output_video_file = output_folder + str(base_name) + "_fmov2_rle_out.avi"
            output_image_file = output_folder + base_name + '_rle_combined_image.png'

            try:
                W, H, F, O, L, frame_data = read_ground_truth(input_rle_txt_file)

                # Create a VideoWriter object to save the video
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for the video
                video_writer = cv2.VideoWriter(output_video_file, fourcc, 2, (W, H))

                combined_image = np.zeros((H, W), dtype=np.uint8)  # Initialize combined image

                for idx, frame in enumerate(frame_data):
                    run_lengths = frame[2:]  # The lengths of the runs
                    image = np.zeros((H, W), dtype=np.uint8)  # Create an empty image for the current frame
                    fill_image_with_runs(image, run_lengths)

                    # Overlay the current frame onto the combined image
                    combined_image = cv2.bitwise_or(combined_image, image)

                    video_writer.write(image)

                cv2.imwrite(output_image_file, combined_image)
                video_writer.release()
                print(f"Saved combined image as {output_image_file} Video saved as {output_video_file}")

            except Exception as e:
                print(f"An error occurred: {e}")

    print("done...")


