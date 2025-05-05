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

pars = []
pars = np.reshape(bboxes[:, :2] + 0.5 * bboxes[:, 2:], (-1, nsplits, 2)).transpose((0, 2, 1))

# 4 point to (8,2) yapilmis- 176 degerden (bboxes) 22 pars cikmissss???
print("len(bboxes)", len(bboxes))
print("len(pars)", len(pars))
print("pars", pars)

pars = np.reshape(pars, (-1, nsplits))
print("pars2", pars)


rads = np.reshape(np.max(0.5 * bboxes[:, 2:], 1), (-1, nsplits))
pars = np.r_[np.zeros((start_ind * 2, nsplits)), pars]
rads = np.r_[np.zeros((start_ind, nsplits)), rads]

