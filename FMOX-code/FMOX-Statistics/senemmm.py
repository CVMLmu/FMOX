import numpy as np
import os

folder = "./"
seqname = "v_box_GTgamma"
bboxes = np.loadtxt(os.path.join(folder,'gt_bbox',seqname + '.txt'))

print(bboxes)
print(type(bboxes))
