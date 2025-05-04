import os
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
    def __init__(self, nfiles, algname):
        self.av_ious = np.zeros(nfiles)
        self.av_times = []
        self.seqi = 0
        self.algname = algname

    def next(self, seqname, means):
        self.av_ious[self.seqi], self.av_psnr[self.seqi], self.av_ssim[self.seqi] = means
        print('{}: Finished seq {}, avg. TIoU {:.3f}'.format(self.algname,seqname, self.av_ious[self.seqi]))
        self.seqi += 1

    def next_time(self, tm):
        self.av_times.append(tm)

    def close(self):
       print('AVERAGES')
       means = np.nanmean(self.av_ious)
       print('{}: TIoU {:.3f}'.format(self.algname, *means))
       print('{}: time {:.3f} seconds'.format(self.algname, np.nanmean(np.array(self.av_times))))
       return means

#######################################################################################################################

class SequenceScoreTracker:
    def __init__(self, nfrms, algname):
        self.all_ious = {}
        self.algname = algname

    def next_traj(self,kk,gt_traj,est_traj,minor_axis_length):
        ious = calciou(gt_traj, est_traj, minor_axis_length)
        ious2 = calciou(gt_traj, est_traj[:,-1::-1], minor_axis_length)
        iou = np.max([np.mean(ious), np.mean(ious2)])
        self.all_ious[kk] = iou
        return iou

    def report(self, seqname, kk):
        print('{}: Seq {}, frm {}, TIoU {:.3f}'.format(self.algname, seqname, kk, self.all_ious.get(kk, 0)))

    def close(self):
        return np.mean(list(self.all_ious.values())) if self.all_ious else 0

#######################################################################################################################

class GroundTruthProcessorX:
    def __init__(self, seqname, bboxes):

        self.nsplits = 12 if '-12' in seqname else 8

        nfrms = len(bboxes[0]) # fmox len - len(glob.glob(os.path.join(seqpath,'*.png')))
        start_ind = 0
        end_ind = nfrms

        pars = []
        # bounding boxes might be in the format [x, y, width, height].
        pars = np.reshape(bboxes[:,:2] + 0.5*bboxes[:,2:], (-1,self.nsplits,2)).transpose((0,2,1))
        pars = np.reshape(pars,(-1,self.nsplits))
        rads = np.reshape(np.max(0.5*bboxes[:,2:],1), (-1,self.nsplits))
        pars = np.r_[np.zeros((start_ind*2,self.nsplits)),pars]
        rads = np.r_[np.zeros((start_ind,self.nsplits)),rads]

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


def evaluate_on(seqname, fmox_bboxes, efficienttam_bboxes, args, callback=None):

    # files : 	files = np.array(glob.glob(os.path.join(folder, 'imgs/*_GTgamma')))
    #           files.sort()

    gt_bboxes = fmox_bboxes
    est_bboxes = efficienttam_bboxes

    # av_score_tracker = AverageScoreTracker(files.shape, args.method_name)
    av_score_tracker = AverageScoreTracker(len(gt_bboxes), args.method_name)

    for kkf, ff in enumerate(files):
        gt_gtp = GroundTruthProcessorX(seqname, gt_bboxes)
        est_gtp = GroundTruthProcessorX(seqname, est_bboxes)

        seq_score_tracker = SequenceScoreTracker(gt_gtp.nfrms, args.method_name)
        for kk in range(gt_gtp.nfrms):
            # TODO: need to provide box seperately one for gt one for estimated ....
            gt_traj, radius, bbox = gt_gtp.get_trajgt(kk)
            est_traj, est_radius, est_bbox = est_gtp.get_trajgt(kk)

            start = time.time()

            av_score_tracker.next_time(time.time() - start)

            if not est_traj is None:
                iou = seq_score_tracker.next_traj(kk, gt_traj, est_traj, radius)

            if args.verbose:
                seq_score_tracker.report(gt_gtp.seqname, kk)

        means = seq_score_tracker.close()
        av_score_tracker.next(gt_gtp.seqname, means)

        if callback:
            callback(kkf, means)

    return av_score_tracker.close()

