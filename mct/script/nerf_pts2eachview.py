import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import argparse
import copy
import glob
import math
import os
import shutil

import cv2
import numpy as np
import open3d as o3d
import torch

# path="../../"
# os.chdir(path)
print(os.getcwd())

import argparse
import collections
import os
import struct

import numpy as np

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )
class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS])

def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model, width=width, height=height, params=params)
    return cameras

def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


"""num_tiles: in x/y direction"""

def pts_image_intersect(pts, img_pose, img_K, width, height):
    c2w = img_pose[:3, :]
    vertex_world = np.asarray(pts.points).transpose()
    ones=np.ones((1,vertex_world.shape[1]))
    vertex_world=np.concatenate([vertex_world,ones],axis=0)
    vertex_pix = img_K @ c2w @ vertex_world
    vertex_pix[0, :] = vertex_pix[0, :] / vertex_pix[2, :]
    vertex_pix[1, :] = vertex_pix[1, :] / vertex_pix[2, :]
    vertex_pix[2, :] = vertex_pix[2, :] / vertex_pix[2, :]

    ind1=vertex_pix[0,:]>0
    ind2=vertex_pix[0,:]<width
    ind3=vertex_pix[1,:]>0
    ind4=vertex_pix[1,:]<height
    ind=ind1*ind2*ind3*ind4
    pts_viewed=vertex_world[:3,ind].transpose()
    return pts_viewed

#colmap_dir: contains scene_bbox.txt
#off for pts
def split_pts_each_view(pts_path,colmap_dir,offx=0,offy=0,offz=0):
    pts=o3d.io.read_point_cloud(pts_path)
    aabb=np.loadtxt(os.path.join(colmap_dir,'sparse/scene_bbox.txt'))

    #frist crop with shifted aabb, then shift back the crop pts
    shift=np.array([offx,offy,offz])
    min_bound=aabb[:3]-shift
    max_bound=aabb[3:]-shift
    aabb=o3d.geometry.AxisAlignedBoundingBox(min_bound,max_bound)
    pts_aoi=pts.crop(aabb).translate((offx,offy,offz))

    poses = []
    in_dir=colmap_dir
    cams = read_cameras_text(os.path.join(in_dir, "sparse/cameras.txt"))
    imgs = read_images_text(os.path.join(in_dir, "sparse/images.txt"))

    poses = []
    heights = []
    widths = []
    Ks = []
    img_names = []
    for _id, img in imgs.items():
        cam = cams[img.camera_id]
        pose = torch.cat([torch.tensor(img.qvec2rotmat()), torch.tensor(img.tvec.reshape(3, 1))], dim=1)
        pose = torch.cat([pose, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)
        poses.append(pose)
        heights.append(torch.tensor(cam.height))
        widths.append(torch.tensor(cam.width))
        Ks.append(torch.tensor([[cam.params[0], 0, cam.params[2]], [0, cam.params[1], cam.params[3]], [0, 0, 1]]))
        img_names.append(img.name)

    poses = torch.stack(poses).float()
    widths = torch.stack(widths).int()
    heights = torch.stack(heights).int()
    Ks = torch.stack(Ks).float()

    for img_id in range(poses.shape[0]):
        pts_viewed = pts_image_intersect(
            pts_aoi,
            poses[img_id, :, :].numpy(),
            Ks[img_id, :, :].numpy(),
            widths[img_id].numpy(),
            heights[img_id].numpy(),
        )
        np.savetxt(os.path.join(colmap_dir,'images/'+img_names[img_id][:-4]+'.txt'),pts_viewed)

split_pts_each_view(r'J:\data\Dortmund\comparison\metashape\dense_point_cloud_center.ply',
                    r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\block_center_small\0\dense',
                    393500,5708000,0)
        




