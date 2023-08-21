import collections
import copy
import enum
import json
import math
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

import read_write_model as colmap

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


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def euler2mat(euler):
    omega = euler[0]
    phi = euler[1]
    kappa = euler[2]
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(kappa) * np.cos(phi)
    R[0, 1] = np.cos(omega) * np.sin(kappa) + np.cos(kappa) * np.sin(omega) * np.sin(phi)
    R[0, 2] = np.sin(kappa) * np.sin(omega) - np.cos(kappa) * np.cos(omega) * np.sin(phi)
    R[1, 0] = np.cos(phi) * np.sin(kappa)
    R[1, 1] = np.sin(kappa) * np.sin(omega) * np.sin(phi) - np.cos(kappa) * np.cos(omega)
    R[1, 2] = -np.cos(kappa) * np.sin(omega) - np.cos(omega) * np.sin(kappa) * np.sin(phi)
    R[2, 0] = -np.sin(phi)
    R[2, 1] = np.cos(phi) * np.sin(omega)
    R[2, 2] = -np.cos(omega) * np.cos(phi)
    return R




class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


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

def colmap2nerf_trans(q,t,dis2m=1.0):
    r_cam_3 = qvec2rotmat(q)
    r_cam=np.identity(4)
    r_cam[:3,:3]=r_cam_3
    trans_cam=np.identity(4)
    trans_cam[0,3]=t[0]*dis2m
    trans_cam[1,3]=t[1]*dis2m
    trans_cam[2,3]=t[2]*dis2m
    r=R.from_euler('x', 180, degrees=True)
    r_cam2cam_3=r.as_matrix()
    r_cam2cam=np.identity(4)
    r_cam2cam[:3,:3]=r_cam2cam_3
    R_cam=trans_cam@r_cam
    R_cam=r_cam2cam@R_cam
    R_world=np.linalg.inv(R_cam)
    return R_world


def colmap2nerf(colmap_folder,nerf_folder):
    a=1
    train_imgs = colmap.read_images_text(colmap_folder+"\images.txt")
    train_cams=colmap.read_cameras_text(colmap_folder+"\cameras.txt")
    train_json={}
    train_json['frames']=[]

    for _, img in train_imgs.items():
        cam = train_cams[img.camera_id]
        height=cam.height
        width=cam.width
        fx=cam[4][0]
        fy=cam[4][1]
        cx=cam[4][2]
        cy=cam[4][3]

        Rt_nerf=colmap2nerf_trans(img.qvec,img.tvec)

        frame_json={}
        frame_json['file_path']="rgb/"+os.path.basename(img.name)
        frame_json['fl_x']=fx
        frame_json['fl_y']=fy
        frame_json['cx']=cx
        frame_json['cy']=cy
        frame_json['w']=width
        frame_json['h']=height
        frame_json['transform_matrix']=[[Rt_nerf[0,0],Rt_nerf[0,1],Rt_nerf[0,2],Rt_nerf[0,3]],
                                        [Rt_nerf[1,0],Rt_nerf[1,1],Rt_nerf[1,2],Rt_nerf[1,3]],
                                        [Rt_nerf[2,0],Rt_nerf[2,1],Rt_nerf[2,2],Rt_nerf[2,3]],
                                        [Rt_nerf[3,0],Rt_nerf[3,1],Rt_nerf[3,2],Rt_nerf[3,3]]]
        train_json['frames'].append(frame_json)

    json_object = json.dumps(train_json, indent=4)
    with open(os.path.join(nerf_folder,'transforms.json'), "w") as outfile:
        outfile.write(json_object)

def colmap2nerf1(colmap_folder,nerf_folder):

    import shutil
    a=1
    train_imgs = colmap.read_images_text(colmap_folder+"\images.txt")
    train_cams=colmap.read_cameras_text(colmap_folder+"\cameras.txt")
    train_json={}
    train_json['frames']=[]

    for _, img in train_imgs.items():
        cam = train_cams[img.camera_id]
        height=cam.height
        width=cam.width
        fx=cam[4][0]
        fy=cam[4][1]
        cx=cam[4][2]
        cy=cam[4][3]

        Rt_nerf=colmap2nerf_trans(img.qvec,img.tvec)

        imgname=os.path.basename(img.name)

        shutil.copyfile(img.name,os.path.join(nerf_folder,"rgb",imgname))

        frame_json={}
        frame_json['file_path']="rgb/"+img.name
        frame_json['fl_x']=fx
        frame_json['fl_y']=fy
        frame_json['cx']=cx
        frame_json['cy']=cy
        frame_json['w']=width
        frame_json['h']=height
        frame_json['transform_matrix']=[[Rt_nerf[0,0],Rt_nerf[0,1],Rt_nerf[0,2],Rt_nerf[0,3]],
                                        [Rt_nerf[1,0],Rt_nerf[1,1],Rt_nerf[1,2],Rt_nerf[1,3]],
                                        [Rt_nerf[2,0],Rt_nerf[2,1],Rt_nerf[2,2],Rt_nerf[2,3]],
                                        [Rt_nerf[3,0],Rt_nerf[3,1],Rt_nerf[3,2],Rt_nerf[3,3]]]
        train_json['frames'].append(frame_json)

    json_object = json.dumps(train_json, indent=4)
    with open(os.path.join(nerf_folder,'transforms.json'), "w") as outfile:
        outfile.write(json_object)

def colmap2nerfcamerapath(colmap_folder,nerf_folder):

    import shutil
    a=1
    train_imgs = colmap.read_images_text(colmap_folder+"\images.txt")
    train_cams=colmap.read_cameras_text(colmap_folder+"\cameras.txt")
    train_json={}
    train_json['camera_path']=[]
    train_json['render_height']=1080
    train_json['render_width']=1920
    train_json['camera_type']='perspective'
    train_json['fps']='24'

    for _, img in train_imgs.items():
        cam = train_cams[img.camera_id]
        height=cam.height
        width=cam.width
        fx=cam[4][0]
        fy=cam[4][1]
        cx=cam[4][2]
        cy=cam[4][3]

        Rt_nerf=colmap2nerf_trans(img.qvec,img.tvec)

        #shutil.copyfile(img.name,os.path.join(nerf_folder,"rgb",imgname))

        frame_json={}
        frame_json['camera_to_world']=[Rt_nerf[0,0],Rt_nerf[0,1],Rt_nerf[0,2],Rt_nerf[0,3],
                                        Rt_nerf[1,0],Rt_nerf[1,1],Rt_nerf[1,2],Rt_nerf[1,3],
                                        Rt_nerf[2,0],Rt_nerf[2,1],Rt_nerf[2,2],Rt_nerf[2,3],
                                        Rt_nerf[3,0],Rt_nerf[3,1],Rt_nerf[3,2],Rt_nerf[3,3]]
        frame_json['fov']=50
        frame_json['image_name']=img.name
        train_json['camera_path'].append(frame_json)

    json_object = json.dumps(train_json, indent=4)
    with open(os.path.join(nerf_folder,'camera_path.json'), "w") as outfile:
        outfile.write(json_object)

def colmap2nerfcamerapath_intrinsic(colmap_folder,nerf_folder):

    import shutil
    a=1
    train_imgs = colmap.read_images_text(colmap_folder+"\images.txt")
    train_cams=colmap.read_cameras_text(colmap_folder+"\cameras.txt")
    train_json={}
    train_json['camera_path']=[]
    train_json['render_height']=1080
    train_json['render_width']=1920
    train_json['camera_type']='perspective'
    train_json['fps']='24'

    for _, img in train_imgs.items():
        cam = train_cams[img.camera_id]
        height=cam.height
        width=cam.width
        fx=cam[4][0]
        fy=cam[4][1]
        cx=cam[4][2]
        cy=cam[4][3]

        Rt_nerf=colmap2nerf_trans(img.qvec,img.tvec)

        #shutil.copyfile(img.name,os.path.join(nerf_folder,"rgb",imgname))

        frame_json={}
        frame_json['camera_to_world']=[Rt_nerf[0,0],Rt_nerf[0,1],Rt_nerf[0,2],Rt_nerf[0,3],
                                        Rt_nerf[1,0],Rt_nerf[1,1],Rt_nerf[1,2],Rt_nerf[1,3],
                                        Rt_nerf[2,0],Rt_nerf[2,1],Rt_nerf[2,2],Rt_nerf[2,3],
                                        Rt_nerf[3,0],Rt_nerf[3,1],Rt_nerf[3,2],Rt_nerf[3,3]]
        frame_json['fl_x']=fx
        frame_json['fl_y']=fy
        frame_json['cx']=cx
        frame_json['cy']=cy

        frame_json['image_name']=img.name
        train_json['camera_path'].append(frame_json)

    json_object = json.dumps(train_json, indent=4)
    with open(os.path.join(nerf_folder,'camera_path.json'), "w") as outfile:
        outfile.write(json_object)


#colmap2nerf(r'E:\data\wriva\drone_colmap\aoi',r'J:\xuningli\cross-view\ns\nerfstudio\data\wrive_drone')
#colmap2nerfcamerapath(r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_full_800\dense\sparse',r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_full_800\dense')
#colmap2nerfcamerapath(r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\dense\sparse',r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\dense')
#colmap2nerfcamerapath_intrinsic(r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\dense\sparse',r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\dense')
colmap2nerfcamerapath_intrinsic(r'J:\xuningli\cross-view\ns\nerfstudio\data\geomvs\dense\sparse',r'J:\xuningli\cross-view\ns\nerfstudio\data\geomvs\dense')