import cv2
import numpy as np
from skimage.draw import line_aa

def write_trajectory(Img, traj, color=(0,255,0), zero_thresh=2, max_segment_length=50):
    """
    Draw trajectory on Img from traj points.
    Skip drawing lines between points that are disconnected or invalid.
    Args:
        Img: image (H x W x 3) numpy array to draw on.
        traj: numpy array of shape (2, N), trajectory points [x, y].
        color: BGR tuple for line color.
        zero_thresh: minimum norm threshold for valid points
        max_segment_length: maximum allowed distance between consecutive trajectory points to be connected.
    Returns:
        Img with trajectory lines drawn.
    """
    num_points = traj.shape[1]

    for kk in range(num_points - 1):
        pt0 = traj[:, kk]
        pt1 = traj[:, kk + 1]

        # Check validity of the points
        if (not np.all(np.isfinite(pt0))) or (not np.all(np.isfinite(pt1))):
            # One or both points invalid, skip
            continue

        if (np.linalg.norm(pt0) < zero_thresh) or (np.linalg.norm(pt1) < zero_thresh):
            # One or both points too close to (0,0), skip
            continue

        # Skip points exactly at placeholder values like (1.0, 1.0)
        if np.allclose(pt0, 1.0) or np.allclose(pt1, 1.0):
            continue

        # Skip zero-length segments
        if np.array_equal(pt0, pt1):
            continue

        # Check distance between points to detect gap
        dist = np.linalg.norm(pt1 - pt0)
        if dist > max_segment_length:
            # Gap too big, skip drawing this segment
            continue

        # Prepare parameters for renderTraj as before
        pars = np.c_[pt0, pt1 - pt0][::-1]

        Img = renderTraj(pars, Img, color)

    return Img


def renderTraj(pars, H, color=(0, 255, 0)):
    thickness = 2
    H = H.copy()

    # Check if parms are valid numbers and within image bounds before drawing
    start = pars[:, 0]
    end = pars[:, 0] + pars[:, 1]
    # If parabola coefficient exists:
    if pars.shape[1] > 2:
        end = pars[:, 0] + pars[:, 1] + pars[:, 2]

    # Check any NaN or Inf in start/end
    if (not np.all(np.isfinite(start))) or (not np.all(np.isfinite(end))):
        return H

    # Check if start or end points are outside image or near zero (0,0)
    h, w = H.shape[:2]
    points_to_check = np.vstack([start, end])
    for pt in points_to_check:
        if (pt[0] < 0 or pt[0] >= h or pt[1] < 0 or pt[1] >= w or
            np.linalg.norm(pt) < 1e-3):
            return H  # Skip drawing if invalid/out of bounds/near zero

    if pars.shape[1] == 2:
        pars = np.concatenate((pars, np.zeros((2, 1))), axis=1)
        ns = 2
    else:
        ns = 5
    ns = max(2, ns)
    rangeint = np.linspace(0, 1, ns)

    for idx in range(ns - 1):
        ti0, ti1 = rangeint[idx], rangeint[idx + 1]
        start = pars[:, 0] + pars[:, 1]*ti0 + pars[:, 2]*ti0*ti0
        end = pars[:, 0] + pars[:, 1]*ti1 + pars[:, 2]*ti1*ti1

        start = np.round(start).astype(int)
        end = np.round(end).astype(int)

        # Validate coords inside image bounds
        if (start[0] < 0 or start[0] >= h or start[1] < 0 or start[1] >= w or
            end[0] < 0 or end[0] >= h or end[1] < 0 or end[1] >= w):
            continue

        rr, cc, val = line_aa(start[0], start[1], end[0], end[1])

        # Filter valid indices
        valid = (
            (rr >= 0) & (rr < h) &
            (cc >= 0) & (cc < w)
        )
        rr, cc, val = rr[valid], cc[valid], val[valid]

        for t in range(-thickness, thickness + 1):
            if t == 0:
                rr_t, cc_t, val_t = rr, cc, val
            else:
                rr_t = rr + t
                cc_t = cc
                valid_t = (rr_t >= 0) & (rr_t < h) & (cc_t >= 0) & (cc_t < w)
                rr_t, cc_t, val_t = rr_t[valid_t], cc_t[valid_t], val[valid_t]

                rr_h = rr
                cc_h = cc + t
                valid_h = (rr_h >= 0) & (rr_h < h) & (cc_h >= 0) & (cc_h < w)
                rr_h, cc_h, val_h = rr_h[valid_h], cc_h[valid_h], val[valid_h]

                if H.ndim == 3:
                    H[rr_t, cc_t, 0] = color[0]
                    H[rr_t, cc_t, 1] = color[1]
                    H[rr_t, cc_t, 2] = color[2]
                    H[rr_h, cc_h, 0] = color[0]
                    H[rr_h, cc_h, 1] = color[1]
                    H[rr_h, cc_h, 2] = color[2]
                else:
                    H[rr_t, cc_t] = 255
                    H[rr_h, cc_h] = 255

            if t == 0:
                if H.ndim == 3:
                    for ch in range(3):
                        H[rr_t, cc_t, ch] = np.maximum(H[rr_t, cc_t, ch], (color[ch] * val_t).astype(H.dtype))
                else:
                    H[rr_t, cc_t] = np.maximum(H[rr_t, cc_t], (255 * val_t).astype(H.dtype))
    return H

def draw_legend(image, gt_color, est_color, pos=(10, 30), spacing=20):
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    line_length = 40
    x, y = pos
    # Ground truth line and label
    cv2.line(img, (x, y), (x + line_length, y), gt_color, thickness)
    cv2.putText(img, "Ground Truth Trajectory", (x + line_length + 10, y + 5), font, font_scale,
                gt_color, thickness, cv2.LINE_AA)
    # Estimated line and label below
    y += spacing
    cv2.line(img, (x, y), (x + line_length, y), est_color, thickness)
    cv2.putText(img, "EfficientTam Estimated Trajectory", (x + line_length + 10, y + 5), font, font_scale,
                est_color, thickness, cv2.LINE_AA)
    return img