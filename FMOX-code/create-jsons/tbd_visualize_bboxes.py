import os
import cv2
import numpy as np


def read_ground_truth(gt_file):
    """Read ground truth data from file."""
    with open(gt_file, 'r') as f:
        lines = f.readlines()

    gt_data = []
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            x_coords = list(map(int, lines[i].strip().split()))
            y_coords = list(map(int, lines[i + 1].strip().split()))

            # Each bounding box consists of 4 points (x,y coordinates)
            bbox = []
            for j in range(len(x_coords)):
                if j < len(y_coords):  # Ensure we have both x and y
                    bbox.append((x_coords[j], y_coords[j]))

            gt_data.append(bbox)

    return gt_data


def draw_bounding_box(img, bbox, color=(0, 255, 0), thickness=2):
    """Draw a bounding box on the image."""
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


def main():
    input_dir = r'C:\Users\user\PycharmProjects\Datasets\data_public\TbD\imgs\fall_cube/'
    output_dir = r'C:\Users\user\PycharmProjects\Datasets\data_public\TbD\imgs/output'
    gt_file = os.path.join(input_dir, 'gt.txt')

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read ground truth data
    gt_data = read_ground_truth(gt_file)

    # Process each image
    for i in range(len(gt_data)):
        img_path = os.path.join(input_dir, f'{i:08d}.png')
        if os.path.exists(img_path):
            # Read image
            img = cv2.imread(img_path)

            if img is not None:
                # Draw bounding box
                img_annotated = draw_bounding_box(img.copy(), gt_data[i])

                # Save annotated image
                output_path = os.path.join(output_dir, f'{i:08d}_annotated.png')
                cv2.imwrite(output_path, img_annotated)
                print(f"Processed image {i+1}/{len(gt_data)}: {output_path}")
            else:
                print(f"Could not read image: {img_path}")
        else:
            print(f"Image does not exist: {img_path}")


if __name__ == '__main__':
    main()
