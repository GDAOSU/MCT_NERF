import os

import cv2
import numpy as np


def merge(in_dir,out_dir):
    out_img=np.zeros((4088,3066,3),np.uint8)
    for i in reversed(range(11)):
        img_path=os.path.join(in_dir,"cam0_block"+str(i)+'.jpg')
        if not os.path.exists(img_path):
            continue
        img=cv2.imread(img_path)
        mask=img>20
        out_img[mask]=img[mask]
        cv2.imwrite(os.path.join(out_dir,"{}.jpg".format(i)),out_img)

merge(r'J:\xuningli\cross-view\ns\nerfstudio\renders\dortmund_blocks_2_16\30k_vis',r'J:\xuningli\cross-view\ns\nerfstudio\renders\dortmund_blocks_2_16\30k_vis\new')
