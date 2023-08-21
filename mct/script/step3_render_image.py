import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from nerfstudio.cameras.camera_paths import (
    get_interpolated_camera_path,
    get_path_from_json,
    get_path_from_json_intrinsic,
    get_path_from_json_transform,
    get_spiral_path,
)
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup


def nerf2colmap(nerf_pose):
    c2w_nerf=torch.eye(4)
    c2w_nerf[:3,:]=nerf_pose
    w2c_nerf=torch.linalg.inv(c2w_nerf).numpy()

    r=R.from_euler('x', -180, degrees=True).as_matrix()
    r_cam2cam=np.identity(4)
    r_cam2cam[:3,:3]=r

    w2c=r_cam2cam@w2c_nerf

    return torch.tensor(w2c)

## scene_aabb: [xmin,ymin,zmin,xmax,ymax,zmax], img_pose: 4x4, w2c
def scenebbox_image_intersect(scene_aabb, img_pose, img_K, width, height):
    c2w = img_pose[:3, :]
    vertex_world = np.array(
        [
            [scene_aabb[0], scene_aabb[1], scene_aabb[2], 1],
            [scene_aabb[0], scene_aabb[4], scene_aabb[2], 1],
            [scene_aabb[3], scene_aabb[4], scene_aabb[2], 1],
            [scene_aabb[3], scene_aabb[1], scene_aabb[2], 1],
            [scene_aabb[0], scene_aabb[1], scene_aabb[5], 1],
            [scene_aabb[0], scene_aabb[4], scene_aabb[5], 1],
            [scene_aabb[3], scene_aabb[4], scene_aabb[5], 1],
            [scene_aabb[3], scene_aabb[1], scene_aabb[5], 1],
        ]
    )
    vertex_pix = img_K @ c2w @ vertex_world.transpose()
    vertex_pix[0, :] = vertex_pix[0, :] / vertex_pix[2, :]
    vertex_pix[1, :] = vertex_pix[1, :] / vertex_pix[2, :]
    vertex_pix[2, :] = vertex_pix[2, :] / vertex_pix[2, :]

    bbox_proj = [np.min(vertex_pix[0, :]), np.max(vertex_pix[0, :]), np.min(vertex_pix[1, :]), np.max(vertex_pix[1, :])]
    intersect_bbox = [
        int(max(0, bbox_proj[0])),
        int(min(width - 1, bbox_proj[1])),
        int(max(0, bbox_proj[2])),
        int(min(height - 1, bbox_proj[3])),
    ]
    if intersect_bbox[0] < intersect_bbox[1] and intersect_bbox[2] < intersect_bbox[3]:
        xlen = intersect_bbox[1] - intersect_bbox[0]
        ylen = intersect_bbox[3] - intersect_bbox[2]
        return True, intersect_bbox
        # if xlen * ylen / (width * height) > 0.8:
        #     return True, intersect_bbox
        # else:
        #     return False, intersect_bbox
    else:
        return False, intersect_bbox


### given a camera, find the intersection bbox in pixels for each block
def calculate_intersection_area_inpixel(camera,block_ids, data_dir):
    K=torch.tensor([[camera.fx[0], 0, camera.cx[0]], [0, camera.fy[0], camera.cy[0]], [0, 0, 1]])
    c2w=camera.camera_to_worlds
    w2c=nerf2colmap(c2w)
    height=camera.height[0]
    width=camera.width[0]
    inter_bbox_list=[]
    for block_id in block_ids:
        scene_bbox_file=os.path.join(data_dir,str(block_id)+"/dense/sparse/scene_bbox.txt")
        scene_bbox=np.loadtxt(scene_bbox_file)
        status,inter_bbox=scenebbox_image_intersect(scene_bbox,w2c.numpy(),K.numpy(),width,height)
        inter_bbox_list.append(inter_bbox)
        if status:
            print("block {}: {}".format(block_id,inter_bbox))
    return inter_bbox_list

def merge_rendered_view_each_block(height,width,rendered_each_blocks,rendered_inter_bboxs):
    out_img=np.zeros((height,width,3),np.uint8)
    for id,img in enumerate(rendered_each_blocks):
        inter_bbox=rendered_inter_bboxs[id]
        out_img[inter_bbox[2]:inter_bbox[3]+1,inter_bbox[0]:inter_bbox[1]+1,:]=img
    return out_img

## render the images 
##data_dir: multi-camera tiling datasets
##trained_model_dir and timestamp: refer to step3_generate_pcd.py/generate_pcd()
#camera_path_file: nerfstudio format, the pose should be in nerfstudio coordinates (not colmap coordiate). Refer to colmap_to_nerf.py/colmap2nerfcamerapath_intrinsic()
def render_novel_view_mct(data_dir,trained_model_dir,timestamp,camera_path_file, out_dir):

    data_block_files=glob.glob(os.path.join(data_dir,"*"))
    block_ids=[int(os.path.basename(block_file)) for block_file in data_block_files]
    print("blocks are: {}".format(block_ids))
    #trained_model_config_files=glob.glob(os.path.join(trained_model_dir,"*/mct_mipnerf/0/config.yml"))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open(camera_path_file, "r", encoding="utf-8") as f:
        camera_path = json.load(f)
        cameras = get_path_from_json_intrinsic(camera_path)
        for camera_idx in range(cameras.size):
            camera=cameras[camera_idx]
            intersection_bbox_list=calculate_intersection_area_inpixel(camera,block_ids,data_dir)
            K=torch.tensor([[camera.fx[0], 0, camera.cx[0]], [0, camera.fy[0], camera.cy[0]], [0, 0, 1]])
            c2w=camera.camera_to_worlds
            height=camera.height[0]
            width=camera.width[0]
            rendered_each_block_imgs=[]
            rendered_each_block_inter_bboxs=[]
            #render for each block
            #rendered_img=np.zeros((camera.height[0],camera.width[0],3))
            for id,block_id in reversed(list(enumerate(block_ids))):
                inter_bbox=intersection_bbox_list[id]
                if inter_bbox[0]>=inter_bbox[1] or inter_bbox[2]>=inter_bbox[3]:
                    continue
                config_file=os.path.join(trained_model_dir,str(block_id)+"/mct_mipnerf/"+timestamp+"/config.yml")
                if not os.path.exists(config_file):
                    continue
                _, pipeline, _, _ = eval_setup(Path(config_file),1024,test_mode="test")

                ##apply input2nerf matrix to novel view camera pose
                dataparser_scale=pipeline.datamanager.train_dataparser_outputs.dataparser_scale
                dataparser_transform=pipeline.datamanager.train_dataparser_outputs.dataparser_transform
                r=c2w[:3,:3]
                C=c2w[:3,3]
                transform_r=dataparser_transform[:3,:3]
                transform_t=dataparser_transform[:3,3]
                C=transform_r@C+transform_t
                C*=dataparser_scale
                r=transform_r@r
                c2w_nerf=torch.zeros((3,4))
                c2w_nerf[:3,3]=C
                c2w_nerf[:3,:3]=r
                camera_nerf=Cameras(
                    fx=camera.fx,
                    fy=camera.fy,
                    cx=camera.cx,
                    cy=camera.cy,
                    camera_to_worlds=c2w_nerf,
                    camera_type=camera.camera_type,
                    times=camera.times,
                )

                ## render novel view
                camera_ray_bundle = camera_nerf.generate_rays(camera_indices=0, aabb_box=None)

                with torch.no_grad():
                    ## render only within aoi_bbox
                    xmin=inter_bbox[0]
                    xmax=inter_bbox[1]
                    ymin=inter_bbox[2]
                    ymax=inter_bbox[3]
                    aoi_height=ymax-ymin+1
                    aoi_width=xmax-xmin+1
                    outputs_list =[]
                    for row in range(ymin,ymax+1):
                        ray_bundle=camera_ray_bundle.flatten()[xmin+row*width:xmax+1+row*width]
                        outputs = pipeline.model.forward(ray_bundle=ray_bundle.to(pipeline.device))
                        outputs_list.append(outputs['rgb_fine'].cpu())
                    output=torch.cat(outputs_list).view(aoi_height, aoi_width, -1)
                    # for output_name, outputs_list in outputs_lists.items():
                    #     if not torch.is_tensor(outputs_list[0]):
                    #         # TODO: handle lists of tensors as well
                    #         continue
                    #     outputs[output_name] = torch.cat(outputs_list).view(aoi_height, aoi_width, -1)  # type: ignore
                    #outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                    output_image = output.numpy()*255
                    output_image=output_image.astype(np.uint8)
                    output_image=cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB)
                    # output_image_crop=np.zeros(output_image.shape,np.uint8)
                    # output_image_crop[inter_bbox[2]:inter_bbox[3],inter_bbox[0]:inter_bbox[1],:]=output_image[inter_bbox[2]:inter_bbox[3],inter_bbox[0]:inter_bbox[1],:]
                    cv2.imwrite(os.path.join(out_dir,"cam"+str(camera_idx)+"_block"+str(block_id)+".jpg"),output_image)
                    rendered_each_block_imgs.append(output_image)
                    rendered_each_block_inter_bboxs.append(inter_bbox)
                

            ##merge rendered image each block together
            rendered_img=merge_rendered_view_each_block(height,width,rendered_each_block_imgs,rendered_each_block_inter_bboxs)
            cv2.imwrite(os.path.join(out_dir,"cam"+str(camera_idx))+".jpg",rendered_img)

def render_novel_view_mct_vis(data_dir,trained_model_dir,timestamp,camera_path_file, out_dir):

    data_block_files=glob.glob(os.path.join(data_dir,"*"))
    block_ids=[int(os.path.basename(block_file)) for block_file in data_block_files]
    print("blocks are: {}".format(block_ids))
    #trained_model_config_files=glob.glob(os.path.join(trained_model_dir,"*/mct_mipnerf/0/config.yml"))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open(camera_path_file, "r", encoding="utf-8") as f:
        camera_path = json.load(f)
        cameras = get_path_from_json_intrinsic(camera_path)
        for camera_idx in range(cameras.size):
            camera=cameras[camera_idx]
            intersection_bbox_list=calculate_intersection_area_inpixel(camera,block_ids,data_dir)
            K=torch.tensor([[camera.fx[0], 0, camera.cx[0]], [0, camera.fy[0], camera.cy[0]], [0, 0, 1]])
            c2w=camera.camera_to_worlds
            height=camera.height[0]
            width=camera.width[0]
            rendered_each_block_imgs=[]
            rendered_each_block_inter_bboxs=[]
            #render for each block
            #rendered_img=np.zeros((camera.height[0],camera.width[0],3))
            for id,block_id in reversed(list(enumerate(block_ids))):
                inter_bbox=intersection_bbox_list[id]
                if inter_bbox[0]>=inter_bbox[1] or inter_bbox[2]>=inter_bbox[3]:
                    continue
                config_file=os.path.join(trained_model_dir,str(block_id)+"/mct_mipnerf/"+timestamp+"/config.yml")
                if not os.path.exists(config_file):
                    continue
                _, pipeline, _, _ = eval_setup(Path(config_file),1024,test_mode="test")

                ##apply input2nerf matrix to novel view camera pose
                dataparser_scale=pipeline.datamanager.train_dataparser_outputs.dataparser_scale
                dataparser_transform=pipeline.datamanager.train_dataparser_outputs.dataparser_transform
                r=c2w[:3,:3]
                C=c2w[:3,3]
                transform_r=dataparser_transform[:3,:3]
                transform_t=dataparser_transform[:3,3]
                C=transform_r@C+transform_t
                C*=dataparser_scale
                r=transform_r@r
                c2w_nerf=torch.zeros((3,4))
                c2w_nerf[:3,3]=C
                c2w_nerf[:3,:3]=r
                camera_nerf=Cameras(
                    fx=camera.fx,
                    fy=camera.fy,
                    cx=camera.cx,
                    cy=camera.cy,
                    camera_to_worlds=c2w_nerf,
                    camera_type=camera.camera_type,
                    times=camera.times,
                )

                ## render novel view
                camera_ray_bundle = camera_nerf.generate_rays(camera_indices=0, aabb_box=None)

                with torch.no_grad():
                    ## render only within aoi_bbox
                    xmin=inter_bbox[0]
                    xmax=inter_bbox[1]
                    ymin=inter_bbox[2]
                    ymax=inter_bbox[3]
                    aoi_height=ymax-ymin+1
                    aoi_width=xmax-xmin+1
                    outputs_list =[]
                    for row in range(ymin,ymax+1):
                        ray_bundle=camera_ray_bundle.flatten()[xmin+row*width:xmax+1+row*width]
                        outputs = pipeline.model.forward(ray_bundle=ray_bundle.to(pipeline.device))
                        outputs_list.append(outputs['rgb_fine'].cpu())
                    output=torch.cat(outputs_list).view(aoi_height, aoi_width, -1)
                    # for output_name, outputs_list in outputs_lists.items():
                    #     if not torch.is_tensor(outputs_list[0]):
                    #         # TODO: handle lists of tensors as well
                    #         continue
                    #     outputs[output_name] = torch.cat(outputs_list).view(aoi_height, aoi_width, -1)  # type: ignore
                    #outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                    output_image = output.numpy()*255
                    output_image=output_image.astype(np.uint8)
                    output_image=cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB)
                    output_image_crop=np.zeros((height,width,3),np.uint8)
                    output_image_crop[inter_bbox[2]:inter_bbox[3]+1,inter_bbox[0]:inter_bbox[1]+1,:]=output_image
                    cv2.imwrite(os.path.join(out_dir,"cam"+str(camera_idx)+"_block"+str(block_id)+".jpg"),output_image_crop)
                    rendered_each_block_imgs.append(output_image)
                    rendered_each_block_inter_bboxs.append(inter_bbox)
        

                    



# render_novel_view_mct(r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\blocks',
#                       r'J:\xuningli\cross-view\ns\nerfstudio\outputs\dortmund_metashape_blocks16',
#                       r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\dense\camera_path.json',
#                       r'J:\xuningli\cross-view\ns\nerfstudio\renders\dortmund_blocks16')

#block16
# render_novel_view_mct(r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\blocks_2_16',
#                       r'J:\xuningli\cross-view\ns\nerfstudio\outputs\dortmund_metashape_blocks_2_16',
#                       '0',
#                       r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\dense_2\camera_path.json',
#                       r'J:\xuningli\cross-view\ns\nerfstudio\renders\dortmund_blocks_2_16\5k')


# render_novel_view_mct_vis(r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\blocks_2_16',
#                       r'J:\xuningli\cross-view\ns\nerfstudio\outputs\dortmund_metashape_blocks_2_16',
#                       '30k',
#                       r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\dense_2\camera_path.json',
#                       r'J:\xuningli\cross-view\ns\nerfstudio\renders\dortmund_blocks_2_16\30k_vis\new')

# render_novel_view_mct(r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\blocks_2_16',
#                       r'J:\xuningli\cross-view\ns\nerfstudio\outputs\dortmund_metashape_blocks_2_16',
#                       '10k',
#                       r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\dense_2\camera_path.json',
#                       r'J:\xuningli\cross-view\ns\nerfstudio\renders\dortmund_blocks_2_16\10k')

#block36
render_novel_view_mct(r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\blocks_2_36',
                      r'J:\xuningli\cross-view\ns\nerfstudio\outputs\dortmund_metashape_blocks_2_36',
                      "30k",
                      r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\dense_2\camera_path.json',
                      r'J:\xuningli\cross-view\ns\nerfstudio\renders\dortmund_blocks_2_36')




