# # import cv2
# # import os
# #
# # def read_all_images_points(filename):
# #     """
# #     Reads a file where each pair of lines corresponds to x and y coordinates
# #     of an image or coordinate set.
# #     Yields a list of (x, y) tuples per image.
# #
# #     Parameters:
# #     filename : str - path to the file
# #
# #     Yields:
# #     List[Tuple[float, float]] - list of points for each image
# #     """
# #     with open(filename, 'r') as f:
# #         while True:
# #             x_line = f.readline()
# #             y_line = f.readline()
# #             if not x_line or not y_line:
# #                 break  # EOF reached or incomplete pair
# #             x_values = list(map(float, x_line.strip().split()))
# #             y_values = list(map(float, y_line.strip().split()))
# #
# #             if len(x_values) != len(y_values):
# #                 raise ValueError("Number of x and y coordinates do not match")
# #
# #             points = list(zip(x_values, y_values))
# #             yield points
# #
# # # Example usage
# # if __name__ == "__main__":
# #     filename = "pars.txt"  # Change this to your file path
# #     img_path = "../fmo_data_extracted_files/Falling_Object/imgs/v_box_GTgamma/"
# #     image_start_name = "00000027.png"
# #
# #     # Read points from the file
# #     all_points = list(read_all_images_points(filename))  # Convert generator to list
# #     # Determine the starting index based on the image_start_name
# #     start_index = int(image_start_name.split('.')[0])  # Extract the index from the filename
# #     # Iterate through all points starting from the determined index
# #     for i in range(start_index, len(all_points)):
# #         points = all_points[i]
# #         image_name = f"{i:08d}.png"  # Assuming image names are zero-padded
# #         image_path = os.path.join(img_path, image_name)
# #
# #         print("image_path", image_path)
# #
# #         # Check if the image file exists
# #         if os.path.isfile(image_path):
# #             img = cv2.imread(image_path)
# #
# #             # Draw points on the image
# #             for x, y in points:
# #                 cv2.circle(img, (x, y), radius=5, color=(0, 255, 0), thickness=-1)  # Green points
# #
# #             # Display the image with points
# #             cv2.imshow(f"Image {i}", img)
# #             cv2.waitKey(0)  # Wait for a key press to close the image window
# #         else:
# #             print(f"Image {image_name} not found.")
# #         # cv2.destroyAllWindows()
#
# import cv2
# import os
#
# def read_all_images_points(filename):
#     """
#     Reads a file where each pair of lines corresponds to x and y coordinates
#     of an image or coordinate set.
#     Yields a list of (x, y) tuples per image.
#
#     Parameters:
#     filename : str - path to the file
#
#     Yields:
#     List[Tuple[float, float]] - list of points for each image
#     """
#     with open(filename, 'r') as f:
#         while True:
#             x_line = f.readline()
#             y_line = f.readline()
#             if not x_line or not y_line:
#                 break  # EOF reached or incomplete pair
#             x_values = list(map(float, x_line.strip().split()))
#             y_values = list(map(float, y_line.strip().split()))
#
#             if len(x_values) != len(y_values):
#                 raise ValueError("Number of x and y coordinates do not match")
#
#             points = list(zip(x_values, y_values))
#             yield points
#
# # Example usage
# if __name__ == "__main__":
#     filename = "pars.txt"  # Change this to your file path
#     img_path = "../fmo_data_extracted_files/Falling_Object/imgs/v_box_GTgamma/"
#     image_start_name = "00000027.png"
#
#     # Read points from the file
#     all_points = list(read_all_images_points(filename))  # Convert generator to list
#
#     # Debug: Print the number of images and points read
#     print(f"Total images with points: {len(all_points)}")
#     print("all_points", all_points)
#
#     # Determine the starting index based on the image_start_name
#     start_index = int(image_start_name.split('.')[0]) - 1  # Extract the index from the filename
#     print(range(start_index, len(all_points)))
#
#     # Iterate through all points starting from the determined index
#     for i in range(start_index, len(all_points)):
#         print(i)
#         points = all_points[i]
#         image_name = f"{i:08d}.png"  # Assuming image names are zero-padded
#         image_path = os.path.join(img_path, image_name)
#
#         # Debug: Print the image path being checked
#         print("Checking image path:", image_path)
#
#         # Check if the image file exists
#         if os.path.isfile(image_path):
#             img = cv2.imread(image_path)
#
#             # Debug: Print the points being drawn
#             print(f"Drawing points on image {image_name}: {points}")
#
#             # Draw points on the image
#             for x, y in points:
#                 cv2.circle(img, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)  # Green points
#
#             # Display the image with points
#             cv2.imshow(f"Image {i}", img)
#             cv2.waitKey(0)  # Wait for a key press to close the image window
#         else:
#             print(f"Image {image_name} not found.")
#
#     cv2.destroyAllWindows()
#

import cv2
import os

import cv2
import os

def read_all_images_points(filename):
    """
    Reads a file where each pair of lines corresponds to x and y coordinates
    of an image or coordinate set.
    Yields a list of (x, y) tuples per image.

    Parameters:
    filename : str - path to the file

    Yields:
    List[Tuple[float, float]] - list of points for each image
    """
    with open(filename, 'r') as f:
        while True:
            x_line = f.readline()
            y_line = f.readline()
            if not x_line or not y_line:
                break  # EOF reached or incomplete pair
            x_values = list(map(float, x_line.strip().split()))
            y_values = list(map(float, y_line.strip().split()))

            if len(x_values) != len(y_values):
                raise ValueError("Number of x and y coordinates do not match")

            points = list(zip(x_values, y_values))
            yield points

if __name__ == "__main__":
    filename = "pars.txt"  # Path to the file with points data
    img_path = "../fmo_data_extracted_files/Falling_Object/imgs/v_box_GTgamma/"
    image_start_name = "00000027.png"  # Starting image filename to begin drawing points

    # Extract the starting index from the image_start_name (assumes zero-padded format)
    start_index = int(image_start_name.split('.')[0])

    # Read all points from the file into a list
    all_points = list(read_all_images_points(filename))

    print(f"Total images with points in file: {len(all_points)}")
    print(f"Starting drawing points from image index: {start_index}")

    for offset, points in enumerate(all_points):
        # Calculate the current image index to load based on start_index + offset
        current_image_index = start_index + offset
        image_name = f"{current_image_index:08d}.png"  # Zero-padded 8-digit format
        image_path = os.path.join(img_path, image_name)

        print(f"Processing Image: {image_name} with {len(points)} points.")

        if not os.path.isfile(image_path):
            print(f"Image file not found: {image_path}. Skipping this image.")
            continue

        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image {image_path}. Skipping.")
            continue

        # Draw all points on the image
        for x, y in points:
            cv2.circle(img, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)  # Green filled circle

        # Display the image with points
        cv2.imshow(f"Image {image_name}", img)
        key = cv2.waitKey(0)  # Wait for key press to proceed to next image
        if key == 27:  # ESC key to exit early
            print("Exit requested by user.")
            break

    cv2.destroyAllWindows()

