import glob
import json
import math
import os

import cv2
import numpy as np


def psnr(gt, predict):
    square=np.square(np.subtract(gt.astype(np.float32),predict.astype(np.float32))).mean(axis=2)
    mask_pre=np.sum(predict,axis=2)
    mask_valid=mask_pre!=0
    mask_invalid=mask_pre==0
    square[mask_invalid]=0
    valid_num=np.sum(mask_valid)
    mse = np.sum(square)/valid_num

    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX) - 10 * np.log10(mse)  


def eval_psnr(predict_dir,gt_dir,cam_json):
    in_data=open(cam_json)
    cam_dict=json.load(in_data)
    cam_list=cam_dict["camera_path"]
    psnr_list=[]
    for id,cam in enumerate(cam_list):
        cam_name=cam["image_name"]
        gt_path=os.path.join(gt_dir,cam_name)
        predict_path=os.path.join(predict_dir,"cam{}.jpg".format(id))
        gt=cv2.imread(gt_path)
        predict=cv2.imread(predict_path)
        psnr_value=psnr(gt,predict)
        print(psnr_value)
        psnr_list.append(psnr_value)
    psnr_arr=np.array(psnr_list)
    psnr_mean=np.mean(psnr_arr)
    print("psnr average: {}".format(psnr_mean))




# eval_psnr(r'J:\xuningli\cross-view\ns\nerfstudio\renders\dortmund_mipnerf_100k',
#           r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\dense\images',
#           r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\dense\camera_path.json')

# eval_psnr(r'J:\xuningli\cross-view\ns\nerfstudio\renders\dortmund_blocks16',
#           r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\dense\images',
#           r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\dense\camera_path.json')

gt=cv2.imread(r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\dense_2\dense\images\005_009_163000210.jpg')
predict=cv2.imread(r'J:\xuningli\cross-view\ns\nerfstudio\renders\dortmund_blocks_2_16\30k_vis\new\10.jpg')
gt=gt[3462:4062,2054:3030,:]
predict=predict[3462:4062,2054:3030,:]
cv2.imwrite(r'J:\xuningli\cross-view\ns\nerfstudio\renders\dortmund_dense_2_mipnerf\mat\ours.png',predict)
a=psnr(gt,predict)
print(a)

predict=cv2.imread(r'J:\xuningli\cross-view\ns\nerfstudio\renders\dortmund_dense_2_mipnerf\10k\cam0.jpg')
predict=predict[3462:4062,2054:3030,:]
cv2.imwrite(r'J:\xuningli\cross-view\ns\nerfstudio\renders\dortmund_dense_2_mipnerf\mat\10k.png',predict)
a=psnr(gt,predict)
print(a)

predict=cv2.imread(r'J:\xuningli\cross-view\ns\nerfstudio\renders\dortmund_dense_2_mipnerf\20k\cam0.jpg')
predict=predict[3462:4062,2054:3030,:]
cv2.imwrite(r'J:\xuningli\cross-view\ns\nerfstudio\renders\dortmund_dense_2_mipnerf\mat\20k.png',predict)
a=psnr(gt,predict)
print(a)

predict=cv2.imread(r'J:\xuningli\cross-view\ns\nerfstudio\renders\dortmund_dense_2_mipnerf\60k\cam0.jpg')
predict=predict[3462:4062,2054:3030,:]
cv2.imwrite(r'J:\xuningli\cross-view\ns\nerfstudio\renders\dortmund_dense_2_mipnerf\mat\60k.png',predict)
a=psnr(gt,predict)
print(a)