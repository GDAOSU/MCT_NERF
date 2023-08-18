import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from nerfstudio.cameras.camera_paths import get_path_from_json_intrinsic
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


def render_novel_view(config_file,camera_path_file, out_dir):

 
    #trained_model_config_files=glob.glob(os.path.join(trained_model_dir,"*/mct_mipnerf/0/config.yml"))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open(camera_path_file, "r", encoding="utf-8") as f:
        camera_path = json.load(f)
        cameras = get_path_from_json_intrinsic(camera_path)
        for camera_idx in range(cameras.size):
            camera=cameras[camera_idx]
            K=torch.tensor([[camera.fx[0], 0, camera.cx[0]], [0, camera.fy[0], camera.cy[0]], [0, 0, 1]])
            c2w=camera.camera_to_worlds
            height=camera.height[0]
            width=camera.width[0]
            #render for each block
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
                num_rays_per_chunk = pipeline.model.config.eval_num_rays_per_chunk
                image_height, image_width = camera_ray_bundle.origins.shape[:2]
                num_rays = len(camera_ray_bundle)
                rgb_list=[]
                for i in range(0, num_rays, num_rays_per_chunk):
                    start_idx = i
                    end_idx = i + num_rays_per_chunk
                    ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                    outputs = pipeline.model.forward(ray_bundle=ray_bundle.to(pipeline.device))
                    rgb_list.append(outputs['rgb_fine'].cpu())
                output=torch.cat(rgb_list).view(image_height, image_width, -1)
                output_image = output*255
                output_image=output_image.numpy().astype(np.uint8)
                output_image=cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB)
                # output_image_crop=np.zeros(output_image.shape,np.uint8)
                # output_image_crop[inter_bbox[2]:inter_bbox[3],inter_bbox[0]:inter_bbox[1],:]=output_image[inter_bbox[2]:inter_bbox[3],inter_bbox[0]:inter_bbox[1],:]
                cv2.imwrite(os.path.join(out_dir,"cam"+str(camera_idx)+".jpg"),output_image)


                    



# render_novel_view(r'./outputs/dortmund_full_mipnerf/0/mct_mipnerf/0/config.yml',
#                   r'./data/dortmund_metashape/dense/camera_path.json',
#                       r'./renders/dortmund_mipnerf_100k')

# render_novel_view(r'./outputs/dortmund_metashape_dense_2_mipnerf/dense_2/mct_mipnerf/5k/config.yml',
#                   r'./data/dortmund_metashape/dense_2/camera_path.json',
#                       r'./renders/dortmund_dense_2_mipnerf/5k')

# render_novel_view(r'./outputs/dortmund_metashape_dense_2_mipnerf/dense_2/mct_mipnerf/10k/config.yml',
#                   r'./data/dortmund_metashape/dense_2/camera_path.json',
#                       r'./renders/dortmund_dense_2_mipnerf/10k')

# render_novel_view(r'./outputs/dortmund_metashape_blocks_2_16/0/mct_mipnerf/10k/config.yml',
#                   r'./data/dortmund_metashape/dense_2/camera_path.json',
#                       r'./renders/dortmund_dense_2_16_bloc')

render_novel_view(r'./outputs/geomvs/0/mct_mipnerf/0/config.yml',
                  r'./data/geomvs/dense/camera_path.json',
                      r'./renders/geomvs')

