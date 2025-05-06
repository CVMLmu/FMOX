import numpy as np


def bbox_to_pars(bbox, num_points=8):
    """
    Convert a bounding box to parameters (points).

    Args:
        bbox: A tuple (min_x, min_y, max_x, max_y) defining the bounding box.
        num_points: Number of points to generate within the bounding box.

    Returns:
        pars: A 2D numpy array of shape (num_points, 2) with generated points.
    """
    min_x, min_y, max_x, max_y = bbox

    # Generate linearly spaced points between min and max for x and y
    x_points = np.linspace(min_x, max_x, num_points)
    y_points = np.linspace(min_y, max_y, num_points)

    # Create a grid of points
    pars = np.array(np.meshgrid(x_points, y_points)).T.reshape(-1, 2)

    return pars


# Example bounding box
bbox = [161, 259, 245, 333]  # (min_x, min_y, max_x, max_y)

# Convert bounding box to parameters
pars = bbox_to_pars(bbox, num_points=6)

# Print the generated parameters
print("Generated Parameters (pars):")
print(pars)
print(len(pars))
