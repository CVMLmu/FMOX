import os
import cv2
import json
import time
import numpy as np
import scipy

"""
def example_usage_with_bboxes():
    # Example bounding boxes (x_min, y_min, x_max, y_max)
    bbox1 = np.array([10, 20, 30, 40])
    bbox2 = np.array([25, 35, 45, 55])

    # Calculate centers of the bounding boxes
    center1 = np.array([(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])
    center2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])

    # Calculate radius as half the diagonal length of bbox1 (assuming both bboxes have approximately same size)
    def bbox_to_radius(bbox):
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return np.sqrt(width ** 2 + height ** 2) / 2

    radius1 = bbox_to_radius(bbox1)
    radius2 = bbox_to_radius(bbox2)

    # For this calciou function we need one radius value, you could take average or max if radii differ
    avg_radius = (radius1 + radius2) / 2

    # Calculate IoU for the circles approximating bbox1 and bbox2
    iou = calciou(center1, center2, avg_radius)

    print(f"Approximate IoU of bounding boxes using calciou circles: {iou:.4f}")
"""

def calc_tiou(gt_traj, traj, rad):
    ns = gt_traj.shape[1]
    est_traj = np.zeros(gt_traj.shape)
    if traj.shape[0] == 4:
        for ni, ti in zip(range(ns), np.linspace(0, 1, ns)):
            est_traj[:, ni] = traj[[1, 0]] * (1 - ti) + ti * traj[[3, 2]]
    else:
        bline = (np.abs(traj[3] + traj[7]) > 1.0).astype(float)
        if bline:
            len1 = np.linalg.norm(traj[[5, 1]])
            len2 = np.linalg.norm(traj[[7, 3]])
            v1 = traj[[5, 1]] / len1
            v2 = traj[[7, 3]] / len2
            piece = (len1 + len2) / (ns - 1)
            for ni in range(ns):
                est_traj[:, ni] = traj[[4, 0]] + np.min([piece * ni, len1]) * v1 + np.max([0, piece * ni - len1]) * v2
        else:
            for ni, ti in zip(range(ns), np.linspace(0, 1, ns)):
                est_traj[:, ni] = traj[[4, 0]] + ti * traj[[5, 1]] + ti * ti * traj[[6, 2]]

    est_traj2 = est_traj[:, -1::-1]

    ious = calciou(gt_traj, est_traj, rad)
    ious2 = calciou(gt_traj, est_traj2, rad)
    return np.max([np.mean(ious), np.mean(ious2)])


def calciou2(p1, p2, rad):
    # Check for zero radius to avoid division by zero
    if rad <= 0:
        raise ValueError("Radius must be greater than zero.")

    # Calculate the distance between points p1 and p2
    dists = np.sqrt(np.sum(np.square(p1 - p2), axis=0))

    # Clamp distances to a maximum of 2 * rad
    dists[dists > 2 * rad] = 2 * rad

    # Calculate normalized distances
    normalized_dists = dists / (2 * rad)
    normalized_dists = np.clip(normalized_dists, -1, 1)  # Clamp values to avoid invalid input

    # Calculate theta
    theta = 2 * np.arccos(normalized_dists)

    # Calculate area and IoU
    A = ((rad * rad) / 2) * (theta - np.sin(theta))
    I = 2 * A
    U = 2 * np.pi * rad * rad - I

    # Avoid division by zero for IoU calculation
    if np.any(U == 0):
        raise ValueError("The union area is zero, cannot compute IoU.")

    iou = I / U
    return iou


def calciou(p1, p2, rad):
    dists = np.sqrt( np.sum( np.square(p1 - p2),0) )
    dists[dists > 2*rad] = 2*rad

    theta = 2*np.arccos( dists/ (2*rad) )
    A = ((rad*rad)/2) * (theta - np.sin(theta))
    I = 2*A
    U = 2* np.pi * rad*rad - I
    iou = I / U
    return iou

#######################################################################################################################

class AverageScoreTracker:
    def __init__(self, nfiles):
        self.av_ious = np.zeros(nfiles)
        self.av_times = []
        self.seqi = 0

    def next(self, seqname, means):
        self.av_ious[self.seqi] = means
        print('AverageScoreTracker, Finished seq {}, avg. TIoU {:.3f}'.format(seqname, self.av_ious[self.seqi]))
        self.seqi += 1

    def next_time(self, tm):
        self.av_times.append(tm)

    def close(self):
       # print('AVERAGES')
       means = np.nanmean(self.av_ious)
       # print("means", means)
       # print('TIoU {:.3f}'.format(means))
       # print('time {:.3f} seconds'.format(np.nanmean(np.array(self.av_times))))
       return means

#######################################################################################################################

class SequenceScoreTracker:
    def __init__(self, nfrms):
        self.all_ious = {}

    def next_traj(self,kk,gt_traj,est_traj,minor_axis_length):
        ious = calciou(gt_traj, est_traj, minor_axis_length)
        ious2 = calciou(gt_traj, est_traj[:,-1::-1], minor_axis_length)
        iou = np.max([np.mean(ious), np.mean(ious2)])
        self.all_ious[kk] = iou
        return iou

    def report(self, seqname, kk):
        print('Seq {}, frm {}, TIoU {:.3f}'.format(seqname, kk, self.all_ious.get(kk, 0)))

    def close(self):
        return np.mean(list(self.all_ious.values())) if self.all_ious else 0

#######################################################################################################################
# def renders2traj(renders,device):
#     masks = renders[:,:,-1]
#     sumx = torch.sum(masks,-2)
#     sumy = torch.sum(masks,-1)
#     cenx = torch.sum(sumy*torch.arange(1,sumy.shape[-1]+1)[None,None].float().to(device),-1) / torch.sum(sumy,-1)
#     ceny = torch.sum(sumx*torch.arange(1,sumx.shape[-1]+1)[None,None].float().to(device),-1) / torch.sum(sumx,-1)
#     est_traj = torch.cat((cenx.unsqueeze(-1),ceny.unsqueeze(-1)),-1)
#     return est_traj

def interpolate_points(points, increment=0.15, num_intermediate=6):
    """
    Given an array of points (N x 2), returns a list where each element corresponds to a pair of consecutive points,
    containing the start point, intermediate points, and the end point. A new point is added that is increment units
    more than the last point in `points`.

    Parameters:
    points : array-like of shape (N, 2)
    increment : float, the amount to add to the last point to create a new endpoint
    num_intermediate : int, number of intermediate points between consecutive points

    Returns:
    List of numpy arrays of shape (num_intermediate + 2, 2)
    Each array corresponds to interpolated points between one pair of points.
    """
    points = np.array(points)
    interpolated_segments = []

    # Interpolate between consecutive points
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        # Create linear interpolation including start and end
        t_values = np.linspace(0, 1, num_intermediate + 2)
        segment_points = np.array([start * (1 - t) + end * t for t in t_values])
        interpolated_segments.append(segment_points)

    # Calculate the new point based on the last point
    if len(points) > 0:
        last_point = points[-1]
        # Create a new point that is increment units more than the last point
        new_point = last_point + np.array([increment, increment])  # Adjust as needed for your use case

        # Interpolate between the last point and the new point
        t_values = np.linspace(0, 1, num_intermediate + 2)
        segment_points = np.array([last_point * (1 - t) + new_point * t for t in t_values])
        interpolated_segments.append(segment_points)

    return np.array(interpolated_segments)

def interpolate_radii1(radii, num_intermediate=6):
    """
    Given an array of radii (N,),
    returns a 2D array where each row corresponds to a pair of consecutive radii,
    containing the start radius, intermediate radii, and the end radius.

    Parameters:
    radii : array-like of shape (N,)
    num_intermediate : int, number of intermediate radii between consecutive radii

    Returns:
    numpy array of shape (N-1, num_intermediate + 2)
    Each row corresponds to interpolated radii between one pair of radii.
    """
    radii = np.array(radii)
    interpolated_segments = []

    for i in range(len(radii) - 1):
        start = radii[i]
        end = radii[i + 1]
        # Create linear interpolation including start and end
        t_values = np.linspace(0, 1, num_intermediate + 2)
        segment_radii = np.array([start * (1 - t) + end * t for t in t_values])
        interpolated_segments.append(segment_radii)

    # Last segment interpolation
    last_start = radii[-1]
    last_end = last_start + 0.15 * (num_intermediate + 1)
    t_values = np.linspace(0, 1, num_intermediate + 2)
    last_segment = np.array([last_start * (1 - t) + last_end * t for t in t_values])
    interpolated_segments.append(last_segment)

    return np.array(interpolated_segments)


class GroundTruthProcessorX:
    def __init__(self, seqname, bboxes, start_ind):
        if '-12' in seqname:
            self.nsplits = 12
        else:
            self.nsplits = 8

        nfrms = len(bboxes) # fmox len - len(glob.glob(os.path.join(seqpath,'*.png')))
        end_ind = nfrms

        pars = []
        bboxes = np.array(bboxes)
        bboxes = bboxes.astype(float)

        # Calculate center coordinates from bounding boxes
        centers = []
        for bbox in bboxes:
            x_center = bbox[0] + bbox[2] / 2  # x_min + width/2
            y_center = bbox[1] + bbox[3] / 2  # y_min + height/2
            centers.append((x_center, y_center))
        # Output the estimated trajectory
        # print("Estimated Trajectory (centers):")
        # Convert to standard Python floats
        centers_as_floats = [[float(center[0]), float(center[1])] for center in centers]
        centers_as_floats = np.array(centers_as_floats)

        pars = interpolate_points(centers_as_floats, 0.15, (self.nsplits-2))
        pars = pars.transpose((0, 2, 1))
        pars = np.reshape(pars, (-1, self.nsplits))

        # ------------------------------------------------------------------------
        # pars = np.reshape(bboxes[:,:2] + 0.5*bboxes[:,2:], (-1,self.nsplits,2)).transpose((0,2,1)) # original
        # pars = np.reshape(pars,(-1,self.nsplits))  # original
        # rads = np.reshape(np.max(0.5 * bboxes[:, 2:], 1), (-1, self.nsplits))  # original

        rads = np.max(0.5 * bboxes[:, 2:], 1)  # == np.maximum(bboxes[:, 2], bboxes[:, 3]) / 2.0
        # radii = np.array([32.0, 32.0, 31.5, 32.5, 32.5, 32.5, 37.5, 41.0, 33.0, 48.0, 46.0, 47.0, 37.0, 36.0,
        #                   35.5, 35.0, 33.0, 32.0, 32.5, 32.0, 31.0, 31.0])
        interpolated_radii = interpolate_radii1(rads, num_intermediate=(self.nsplits-2))

        if seqname == "HighFPS_GT_depth2":
            np.savetxt('HighFPS_GT_depth2_interpolated_radii.txt', interpolated_radii, fmt='%.3f', delimiter=' ')
            np.savetxt('HighFPS_GT_depth2_pars.txt', pars, fmt='%.3f', delimiter=' ')

        pars = np.r_[np.zeros((start_ind * 2, self.nsplits)), pars]
        rads = np.r_[np.zeros((start_ind, self.nsplits)), interpolated_radii]

        self.pars = pars
        self.rads = rads
        self.start_ind = start_ind
        self.nfrms = nfrms
        self.seqname = seqname
        print('Sequence {} has {} frames'.format(seqname, nfrms))

    def get_trajgt(self, kk):
        par = self.pars[2 * (kk + self.start_ind):2 * (kk + self.start_ind + 1), :].T
        self.nsplits = par.shape[0]
        parsum = par.sum(1)
        nans = np.isnan(parsum)
        if nans.sum() > 0:
            ind = np.nonzero(nans)[0]
            for indt in ind:
                if indt == 0:
                    par[indt, :] = par[np.nonzero(~nans)[0][0], :]
                elif indt < self.nsplits - 1 and not nans[indt + 1]:
                    par[indt, :] = (par[indt - 1, :] + par[indt + 1, :]) / 2
                else:
                    par[indt, :] = par[indt - 1, :]

        bbox = (par[:, 1].min(), par[:, 0].min(), par[:, 1].max(), par[:, 0].max())
        if self.rads.shape[0] > 1:
            radius = np.round(np.nanmax(self.rads[self.start_ind + kk, :])).astype(int)
        else:
            radius = np.round(self.rads[0, 0]).astype(int)
        bbox = np.array(bbox).astype(int)
        return par.T, radius, bbox

import vis_trajectory

def evaluate_on(dataset_name, seqname, original_I, fmox_bboxes, efficienttam_bboxes, start_ind, callback=None):
    gt_bboxes = fmox_bboxes
    est_bboxes = efficienttam_bboxes

    # av_score_tracker = AverageScoreTracker(files.shape, args.method_name)
    av_score_tracker = AverageScoreTracker(len(gt_bboxes))

    # for kkf, ff in enumerate(files):
    # for kkf in range(len(gt_bboxes)):
    for kkf in range(1):
        est_gtp = GroundTruthProcessorX(seqname, est_bboxes, start_ind)
        gt_gtp = GroundTruthProcessorX(seqname, gt_bboxes, start_ind)

        # seq_score_tracker = SequenceScoreTracker(gt_gtp.nfrms, args.method_name)
        seq_score_tracker = SequenceScoreTracker(gt_gtp.nfrms)

        white_img = np.ones(original_I.shape, dtype=original_I.dtype) * 255

        for kk in range(gt_gtp.nfrms):
            gt_traj, radius, bbox = gt_gtp.get_trajgt(kk)
            est_traj, est_radius, est_bbox = est_gtp.get_trajgt(kk)

            start = time.time()

            av_score_tracker.next_time(time.time() - start)

            if not est_traj is None:
                iou = seq_score_tracker.next_traj(kk, gt_traj, est_traj, radius)

            # if args.verbose:
            # seq_score_tracker.report(gt_gtp.seqname, kk)

            # white_img = vis_trajectory.write_trajectory(white_img, gt_traj, (0, 255, 0))
            white_img = vis_trajectory.write_trajectory(white_img, gt_traj, color=(0, 255, 0), zero_thresh=2)
            white_img = vis_trajectory.write_trajectory(white_img, est_traj, color=(0, 0, 255), zero_thresh=2)
            # white_img = vis_trajectory.write_trajectory(white_img, est_traj, (0, 0, 255))
            white_img = vis_trajectory.draw_legend(white_img, (0, 255, 0), (0, 0, 255), pos=(10, 30), spacing=20)

        if kk == len(gt_bboxes) - 1:
            folder_path = "./efficientTAM_traj_vis/"
            cv2.imwrite(folder_path + "efficientTAM_traj_" + str(dataset_name) + "_" + str(seqname) + ".jpg", white_img)
        # cv2.imshow("white_img", white_img)
        # cv2.imshow("original_I", original_I)
        # cv2.waitKey(0)

        means = seq_score_tracker.close()
        # print("gt_gtp.seqname, means", gt_gtp.seqname, means)
        av_score_tracker.next(gt_gtp.seqname, means)

        if callback:
            callback(kkf, means)

    return av_score_tracker.close()

