import glob
import os

import cv2
import numpy as np

musk_thresh=15
def gen_mask(in_dir):
    imgs=glob.glob(os.path.join(in_dir,"*.jpg"))
    for img in imgs:
        img_cv=cv2.imread(img)
        img_musk=(np.sum(img_cv,axis=2)>musk_thresh)*255
        cv2.imwrite(img[:-4]+"_mask.jpg",img_musk)

gen_mask(r"J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\dense_2\dense\images")
    

