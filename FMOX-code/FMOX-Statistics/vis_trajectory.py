import cv2
import numpy as np
from skimage.draw import line_aa

def write_trajectory(Img, traj, color):
    # print("traj range", range(traj.shape[1]-1))
    for kk in range(traj.shape[1]-1):
        Img = renderTraj(np.c_[traj[:,kk], traj[:,kk+1]-traj[:,kk]][::-1], Img, color)
        # print("renderTraj input", np.c_[traj[:,kk], traj[:,kk+1]-traj[:,kk]][::-1])
        # cv2.imshow("xxx", Img)
        # cv2.waitKey(0)
    # Img[traj[1].astype(int),traj[0].astype(int),1] = 1.0
    return Img

def renderTraj(pars, H, color=(0, 255, 0)):
    thickness = 2
    H = H.copy()
    # Input: pars is either 2x2 (line) or 2x3 (parabola)
    if pars.shape[1] == 2:
        pars = np.concatenate((pars, np.zeros((2, 1))), 1)
        ns = 2
    else:
        ns = 5

    ns = np.max([2, ns])

    rangeint = np.linspace(0, 1, ns)
    for timeinst in range(rangeint.shape[0] - 1):
        ti0 = rangeint[timeinst]
        ti1 = rangeint[timeinst + 1]
        start = pars[:, 0] + pars[:, 1] * ti0 + pars[:, 2] * (ti0 * ti0)
        end = pars[:, 0] + pars[:, 1] * ti1 + pars[:, 2] * (ti1 * ti1)
        start = np.round(start).astype(np.int32)
        end = np.round(end).astype(np.int32)

        rr, cc, val = line_aa(start[0], start[1], end[0], end[1])
        valid = np.logical_and(np.logical_and(rr < H.shape[0], cc < H.shape[1]),
                               np.logical_and(rr >= 0, cc >= 0))
        rr = rr[valid]
        cc = cc[valid]
        val = val[valid]

        for t in range(-thickness, thickness + 1):
            if t == 0:
                # main line
                rr_t, cc_t, val_t = rr, cc, val
            else:
                # vertical offsets
                rr_t = rr + t
                cc_t = cc
                valid_t = np.logical_and(np.logical_and(rr_t < H.shape[0], cc_t < H.shape[1]),
                                         np.logical_and(rr_t >= 0, cc_t >= 0))
                rr_t = rr_t[valid_t]
                cc_t = cc_t[valid_t]
                val_t = val[valid_t]

                # horizontal offsets
                rr_h = rr
                cc_h = cc + t
                valid_h = np.logical_and(np.logical_and(rr_h < H.shape[0], cc_h < H.shape[1]),
                                         np.logical_and(rr_h >= 0, cc_h >= 0))
                rr_h = rr_h[valid_h]
                cc_h = cc_h[valid_h]
                val_h = val[valid_h]

                # Draw vertical offset line
                if len(H.shape) > 2:
                    # Assign fixed color channels, independent of val_t (alpha)
                    H[rr_t, cc_t, 0] = color[0]  # Blue
                    H[rr_t, cc_t, 1] = color[1]  # Green
                    H[rr_t, cc_t, 2] = color[2]  # Red
                else:
                    H[rr_t, cc_t] = 255  # For grayscale, set max intensity

                # Draw horizontal offset line
                if len(H.shape) > 2:
                    H[rr_h, cc_h, 0] = color[0]
                    H[rr_h, cc_h, 1] = color[1]
                    H[rr_h, cc_h, 2] = color[2]
                else:
                    H[rr_h, cc_h] = 255

            # Draw main line (or single offset when t==0)
            if t == 0:
                if len(H.shape) > 2:
                    # Use val_t to modulate intensity for anti-aliasing
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