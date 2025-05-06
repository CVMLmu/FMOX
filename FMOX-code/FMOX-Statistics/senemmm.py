import numpy as np
import os

folder = "./"
seqname = "v_box_GTgamma"
bboxes = np.loadtxt(os.path.join(folder,'gt_bbox',seqname + '.txt'))

print(bboxes)
print(type(bboxes))

if '-12' in seqname:
    nsplits = 12
else:
    nsplits = 8

nfrms = len(bboxes)  # fmox len - len(glob.glob(os.path.join(seqpath,'*.png')))
start_ind = 0
end_ind = nfrms
nsplits = 8


kkk = np.reshape(bboxes[:, :2] + 0.5 * bboxes[:, 2:], (-1, nsplits, 2))
print("kkk", kkk)

"""kkk [[[210.5   297.   ]
  [210.65  297.59 ]
  [210.82  299.   ]
  [211.12  299.08 ]
  [210.71  300.55 ]
  [210.92  302.46 ]
  [211.47  304.17 ]
  [211.28  304.62 ]]

 [[211.26  307.1  ]
  [211.12  308.47 ]
  [211.14  311.23 ]
  [210.89  312.75 ]
  [211.47  315.61 ]
  [211.2   317.68 ]
  [211.4   320.1  ]
  [211.28  321.73 ]]
"""

"""
[

[[208.  297.5]
[208.  297.5]
[208.  297.5]
[208.  297.5]
[208.  297.5]
[208.  297.5]
[208.  297.5]
[208.  297.5]
[208.6  308.0]]

[208.9  308.5]
 
 
 [208.5 324.5]
 
 
 [209.5 347. ]
"""


pars = []
pars = np.reshape(bboxes[:, :2] + 0.5 * bboxes[:, 2:], (-1, nsplits, 2)).transpose((0, 2, 1))

# 4 point to (8,2) yapilmis- 176 degerden (bboxes) 22 pars cikmissss???
print("len(bboxes)", len(bboxes))
print("len(pars)", len(pars))
print("pars", pars)

""" pars [[[210.5   210.65  210.82  211.12  210.71  210.92  211.47  211.28 ]  x
  [297.    297.59  299.    299.08  300.55  302.46  304.17  304.62 ]] y

 [[211.26  211.12  211.14  210.89  211.47  211.2   211.4   211.28 ]  x
  [307.1   308.47  311.23  312.75  315.61  317.68  320.1   321.73 ]]  y

 [[211.3   211.4   211.33  211.58  212.03  212.36  212.85  212.39 ]
  [324.75  327.41  329.5   332.59  336.15  339.62  344.16  346.56 ]]

 [[212.85  213.145 212.56  212.74  212.45  212.11  211.72  212.12 ]
  [351.36  354.745 358.38  361.43  363.89  367.53  371.32  375.08 ]]

 [[211.98  212.27  212.44  212.61  212.965 213.855 214.28  214.44 ]
  [377.86  382.09  383.74  388.49  394.15  398.73  403.91  408.49 ]]

"""
pars = np.reshape(pars, (-1, nsplits))
print("pars2", pars)
"""pars2 [[210.5   210.65  210.82  211.12  210.71  210.92  211.47  211.28 ]
 [297.    297.59  299.    299.08  300.55  302.46  304.17  304.62 ]
 [211.26  211.12  211.14  210.89  211.47  211.2   211.4   211.28 ]
 [307.1   308.47  311.23  312.75  315.61  317.68  320.1   321.73 ]
 [211.3   211.4   211.33  211.58  212.03  212.36  212.85  212.39 ]
 [324.75  327.41  329.5   332.59  336.15  339.62  344.16  346.56 ]"""

rads = np.reshape(np.max(0.5 * bboxes[:, 2:], 1), (-1, nsplits))

print("rads 1", rads)
print("rads 1", len(rads))

"""
rads 1 [[38.5   38.5   38.5   38.5   38.5   39.27  38.5   38.5  ]
 [38.5   38.5   38.5   38.5   38.5   38.5   38.5   38.5  ]
 [38.5   38.5   38.5   38.5   39.27  39.27  38.5   39.27 ]
 [39.27  40.855 39.27  38.5   38.5   38.5   38.5   38.5  ]
 [38.5   38.5   38.5   38.5   37.745 37.745 38.5   38.5  ]
 [38.5   38.5   38.5   38.5   38.5   38.5   38.5   38.5  ]
"""

pars = np.r_[np.zeros((start_ind * 2, nsplits)), pars]
rads = np.r_[np.zeros((start_ind, nsplits)), rads]

print("rads 2", rads)
# sss [32.  32.  31.5 32.5 32.5 32.5 37.5 41.  33.  48.  46.  47.  37.  36.
#  35.5 35.  33.  32.  32.5 32.  31.  31. ]
"""rads 2 [[38.5   38.5   38.5   38.5   38.5   39.27  38.5   38.5  ]
 [38.5   38.5   38.5   38.5   38.5   38.5   38.5   38.5  ]
 [38.5   38.5   38.5   38.5   39.27  39.27  38.5   39.27 ]
 [39.27  40.855 39.27  38.5   38.5   38.5   38.5   38.5  ]
 [38.5   38.5   38.5   38.5   37.745 37.745 38.5   38.5  ]
 [38.5   38.5   38.5   38.5   38.5   38.5   38.5   38.5  ]"""

import numpy as np


def interpolate_points(points, num_intermediate=7):
    """
    Given an array of points (N x 2),
    returns a list where each element corresponds to a pair of consecutive points,
    containing the start point, 7 intermediate points, and the end point.

    Parameters:
    points : array-like of shape (N, 2)
    num_intermediate : int, number of intermediate points between consecutive points

    Returns:
    List of numpy arrays of shape (num_intermediate + 2, 2)
    Each array corresponds to interpolated points between one pair of points.
    """
    points = np.array(points)
    interpolated_segments = []
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        # Create linear interpolation including start and end
        # t goes from 0 to 1 with (num_intermediate+2) points
        t_values = np.linspace(0, 1, num_intermediate + 2)
        segment_points = np.array([start * (1 - t) + end * t for t in t_values])
        interpolated_segments.append(segment_points)
    return np.array(interpolated_segments)


def flatten_coords(coords):
    # Convert to numpy array if it is not already
    arr = np.array(coords)
    # Reshape to (-1, 2) since each point has 2 coordinates (x, y)
    flat_coords = arr.reshape(-1, 2)
    return flat_coords



# Example usage
if __name__ == "__main__":
    points = np.array([[208., 297.5],
                       [208., 308.5],
                       [208.5, 324.5],
                       [209.5, 347.],
                       [210.5, 375.5],
                       [210.5, 421.5],
                       [211., 457.5],
                       [212.5, 502.],
                       [213., 543.5],
                       [213.5, 617.],
                       [215., 683.],
                       [215.5, 760.],
                       [218., 816.],
                       [210., 835.],
                       [206.5, 816.5],
                       [202., 807.],
                       [198., 799.],
                       [193., 799.],
                       [189.5, 803.],
                       [185., 813.],
                       [181., 826.5],
                       [179., 829.]])

    result = interpolate_points(points, num_intermediate=7)

    # Print first segment as example
    print("First segment (start, intermediates, end):")
    print(result)
    flat_coords = result.reshape(-1, 2)
    print("flat_coords", flat_coords)



import numpy as np


def calculate_radius(pars, nsplits):
    # Assuming pars is a 2D array where we want to calculate the radius
    # based on the last columns (similar to bboxes[:, 2:])

    # For this example, let's assume we want to take the last 2 columns
    # You can adjust the slicing based on your actual requirements
    selected_columns = pars[:, -2:]  # Adjust this as needed

    # Calculate the radius
    rads = np.max(0.5 * selected_columns, axis=1)

    # Reshape the result
    rads = np.reshape(rads, (-1, nsplits))

    return rads


# Example usage
pars = np.array([
    [208., 208., 208., 208., 208., 208., 208., 208.],
    [297.5, 299.07142857, 300.64285714, 302.21428571, 303.78571429, 305.35714286, 306.92857143, 308.5],
    [208., 208.07142857, 208.14285714, 208.21428571, 208.28571429, 208.35714286, 208.42857143, 208.5],
    [308.5, 310.78571429, 313.07142857, 315.35714286, 317.64285714, 319.92857143, 322.21428571, 324.5],
    [208.5, 208.64285714, 208.78571429, 208.92857143, 209.07142857, 209.21428571, 209.35714286, 209.5]
])

# Define the number of splits
nsplits = 4  # Adjust this based on your requirements

# Calculate the radius
calculated_radius = calculate_radius(pars, nsplits)

# Print the result
print("Calculated Radius:\n", calculated_radius)

