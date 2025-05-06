import numpy as np

def bbox_calciou():
    bboxes = np.array(bboxes)
    bboxes = bboxes.astype(float)
    print("sss", np.maximum(bboxes[:, 2], bboxes[:, 3]) / 2.0)