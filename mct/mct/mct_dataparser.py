# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dortmund dataset parser. Datasets and documentation here: http://phototour.cs.washington.edu/datasets/"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
from rich.progress import Console
from torchtyping import TensorType
from typing_extensions import Literal

from mct import mct_camera_utils, mct_dataparseroutputs
from mct.mct_scene_box import SceneBox
from mct.mct_utils import read_cameras_text, read_images_text
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig

#from nerfstudio.data.scene_box import SceneBox

CONSOLE = Console(width=120)


@dataclass
class MCTDataParserConfig(DataParserConfig):
    """Phototourism dataset parser config"""

    _target: Type = field(default_factory=lambda: MCT)
    """target class to instantiate"""
    data: Path = Path("")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: str = "black"
    """alpha color of background"""
    train_split_percentage: float = 0.9
    """The percent of images to use for training. The remaining images are for eval."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "none"] = "none"
    """The method to use for orientation."""
    auto_scale_poses: bool = False
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    center_poses: bool = False
    """Whether to center the poses."""

@dataclass
class MCT(DataParser):
    """Dortmund dataset. This is based on https://github.com/kwea123/nerf_pl/blob/nerfw/datasets/phototourism.py
    and uses colmap's utils file to read the poses.
    """

    config: MCTDataParserConfig

    def __init__(self, config: MCTDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data

    ## for colmap with k1,k2,k3,p1,p1,near,far params
    def _generate_dataparser_outputs(self, split="train"):
        # BLOCK_MODE = True
        ## whole bbox
        # aoi_bbox = [-80, -85, -90, 80, 61, -75]
        ## building1
        # aoi_bbox = [-18, -13, -87, 1, 5, -80]

        image_filenames = []
        has_mask = False
        mask_filenames = []
        poses = []

        with CONSOLE.status(f"[bold green]Reading dortmund images and poses for {split} split...") as _:
            cams = read_cameras_text(self.data / "dense/sparse/cameras.txt")
            imgs = read_images_text(self.data / "dense/sparse/images.txt")
        aoi_bbox = np.loadtxt(self.data / "dense/sparse/scene_bbox.txt")
        poses = []
        fxs = []
        fys = []
        cxs = []
        cys = []
        heights = []
        widths = []
        Ks = []
        image_filenames = []

        for _id, img in imgs.items():
            # for _id, cam in cams.items():
            # img = imgs[_id]
            cam = cams[img.camera_id]

            pose = torch.cat([torch.tensor(img.qvec2rotmat()), torch.tensor(img.tvec.reshape(3, 1))], dim=1)
            pose = torch.cat([pose, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)
            poses.append(torch.linalg.inv(pose))
            heights.append(torch.tensor(cam.height))
            widths.append(torch.tensor(cam.width))
            fxs.append(torch.tensor(cam.params[0]))
            fys.append(torch.tensor(cam.params[1]))
            cxs.append(torch.tensor(cam.params[2]))
            cys.append(torch.tensor(cam.params[3]))
            Ks.append(torch.tensor([[cam.params[0], 0, cam.params[2]], [0, cam.params[1], cam.params[3]], [0, 0, 1]]))
            image_filenames.append(self.data / "dense/images" / img.name)
            mask_path = self.data / "dense/images" / (img.name[:-4] + "_mask" + img.name[-4:])
            mask_filenames.append(mask_path)

        poses = torch.stack(poses).float()
        poses[..., 1:3] *= -1
        fxs = torch.stack(fxs).float()
        fys = torch.stack(fys).float()
        cxs = torch.stack(cxs).float()
        cys = torch.stack(cys).float()
        widths = torch.stack(widths).int()
        heights = torch.stack(heights).int()
        # distortion_params = torch.stack(distortion_params).float()
        Ks = torch.stack(Ks).float()
        # nears = torch.stack(nears).float()
        # fars = torch.stack(fars).float()

        # filter image_filenames and poses based on train/eval split percentage
        num_images = len(image_filenames)
        num_train_images = math.ceil(num_images * self.config.train_split_percentage)
        num_eval_images = num_images - num_train_images
        i_all = np.arange(num_images)
        i_train = np.linspace(
            0, num_images - 1, num_train_images, dtype=int
        )  # equally spaced training images starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        i_all = torch.tensor(i_all)
        i_train = torch.tensor(i_train)
        i_eval = torch.tensor(i_eval)
        assert len(i_eval) == num_eval_images
        if split == "train":
            indices = i_train.long()
        elif split in ["val", "test"]:
            indices = i_eval.long()
        elif split == "all":
            indices = i_all.long()
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        # poses = camera_utils.auto_orient_and_center_poses(
        #     poses, method=self.config.orientation_method, center_poses=self.config.center_poses
        # )

        (
            xyz_min,
            xyz_max,
            sampling_xyz_min,
            sampling_xyz_max,
            poses,
        ) = mct_camera_utils.center_scale_poses_and_compute_frustum(poses, aoi_bbox)

        # make the scene_bbox a little bit larger
        scene_bbox_expand_scale = 2
        xyz_length = xyz_max - xyz_min
        scene_bbox_expand_margin = xyz_length * (scene_bbox_expand_scale - 1) / 2
        xyz_max[0] += scene_bbox_expand_margin[0]
        xyz_min[0] -= scene_bbox_expand_margin[0]
        xyz_max[1] += scene_bbox_expand_margin[1]
        xyz_min[1] -= scene_bbox_expand_margin[1]
        # xyz_max[2] += scene_bbox_expand_margin[2]
        # xyz_min[2] -= scene_bbox_expand_margin[2]
        sampling_xyz_length = sampling_xyz_max - sampling_xyz_min
        sampling_scene_bbox_expand_margin = sampling_xyz_length * (scene_bbox_expand_scale - 1) / 2
        sampling_xyz_max[0] += sampling_scene_bbox_expand_margin[0]
        sampling_xyz_min[0] -= sampling_scene_bbox_expand_margin[0]
        sampling_xyz_max[1] += sampling_scene_bbox_expand_margin[1]
        sampling_xyz_min[1] -= sampling_scene_bbox_expand_margin[1]
        # sampling_xyz_max[2] += sampling_scene_bbox_expand_margin[2]
        # sampling_xyz_min[2] -= sampling_scene_bbox_expand_margin[2]

        aabb = torch.tensor(
            [[xyz_min[0], xyz_min[1], xyz_min[2]], [xyz_max[0], xyz_max[1], xyz_max[2]]], dtype=torch.float32
        )
        sampling_aabb = torch.tensor(
            [
                [sampling_xyz_min[0], sampling_xyz_min[1], sampling_xyz_min[2]],
                [sampling_xyz_max[0], sampling_xyz_max[1], sampling_xyz_max[2]],
            ],
            dtype=torch.float32,
        )

        scene_box = SceneBox(sampling_aabb)
        sampling_box = SceneBox(sampling_aabb)

        cameras = Cameras(
            camera_to_worlds=poses[:, :3, :4],
            fx=fxs,
            fy=fys,
            cx=cxs,
            cy=cys,
            height=heights,
            width=widths,
            # distortion_params=distortion_params,
            camera_type=CameraType.PERSPECTIVE,
        )

        cameras = cameras[indices]
        image_filenames = [image_filenames[i] for i in indices]

        assert len(cameras) == len(image_filenames)
        if has_mask:
            mask_filenames = [mask_filenames[i] for i in indices]
            dataparser_outputs = mct_dataparseroutputs.MCTDataParserOutputs(
                image_filenames=image_filenames,
                mask_filenames=mask_filenames,
                cameras=cameras,
                scene_box=scene_box,
                sampling_box=sampling_box,
            )
        else:
            dataparser_outputs = mct_dataparseroutputs.MCTDataParserOutputs(
                image_filenames=image_filenames, cameras=cameras, scene_box=scene_box, sampling_box=sampling_box
            )

        return dataparser_outputs
