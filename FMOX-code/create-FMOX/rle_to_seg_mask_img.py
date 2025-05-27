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

    return image

def create_combined_image(W, H, frame_data):
    """Create a single image that combines all annotations."""
    combined_image = np.zeros((H, W), dtype=np.uint8)  # Initialize combined image

    for idx, frame in enumerate(frame_data):
        run_lengths = frame[2:]  # The lengths of the runs
        image = np.zeros((H, W), dtype=np.uint8)  # Create an empty image for the current frame
        image = fill_image_with_runs(image, run_lengths)

        # Overlay the current frame onto the combined image
        combined_image = cv2.bitwise_or(combined_image, image)

    return combined_image

def rle_to_mask_img(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through the files in the specified directory
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            base_name, extension = os.path.splitext(filename)
            input_rle_txt_file = os.path.join(input_folder, filename)
            output_video_file = os.path.join(output_folder, f"{base_name}_fmov2_rle_out.avi")
            output_image_file = os.path.join(output_folder, f"{base_name}_rle_combined_image.png")

            try:
                W, H, F, O, L, frame_data = read_ground_truth(input_rle_txt_file)

                print(f"Video dimensions: {W}x{H}")

                # Create a VideoWriter object to save the video
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Try a different codec
                video_writer = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'MJPG'), 20, (int(W), int(H)), isColor=False)

                if not video_writer.isOpened():
                    print(f"Error: Could not open video writer for {output_video_file}")
                    continue

                combined_image = np.zeros((H, W), dtype=np.uint8)  # Initialize combined image

                for idx, frame in enumerate(frame_data):
                    run_lengths = frame[2:]  # The lengths of the runs
                    image = np.zeros((H, W), dtype=np.uint8)  # Create an empty image for the current frame
                    image2 = fill_image_with_runs(image, run_lengths)
                    print(image2.shape)
                    # Save the current frame as an image for debugging
                    # cv2.imwrite(os.path.join(output_folder, f"{base_name}_frame_{idx}.png"), image)

                    # Check if the image is valid before writing
                    if image2 is not None and image2.size > 0:
                        print(f"Writing frame {idx + 1} with shape: {image2.shape}")
                        video_writer.write(image2)
                    else:
                        print(f"Frame {idx + 1} is empty or invalid.")

                    # Overlay the current frame onto the combined image
                    combined_image = cv2.bitwise_or(combined_image, image)

                # Save the combined image
                cv2.imwrite(output_image_file, combined_image)

                # Release the video writer
                video_writer.release()
                print(f"Saved combined image as {output_image_file} and video as {output_video_file}")

            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")

    print("done...")





