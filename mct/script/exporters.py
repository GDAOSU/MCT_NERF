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


"""Algorithms for exporting data."""

from __future__ import annotations

import math
import os
import sys
from typing import Tuple

import open3d as o3d
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from nerfstudio.configs.base_config import Config  # pylint: disable=unused-import
from nerfstudio.pipelines.base_pipeline import Pipeline

CONSOLE = Console(width=120)


# def generate_point_cloud(
#     pipeline: Pipeline,
#     num_points: int = 1000000,
#     remove_outliers: bool = True,
#     estimate_normals: bool = False,
#     rgb_output_name: str = "rgb",
#     depth_output_name: str = "depth",
#     use_bounding_box: bool = True,
#     bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
#     bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
# ) -> o3d.geometry.PointCloud:
#     """Generate a point cloud from a nerf.

#     Args:
#         pipeline: Pipeline to evaluate with.
#         num_points: Number of points to generate. May result in less if outlier removal is used.
#         remove_outliers: Whether to remove outliers.
#         estimate_normals: Whether to estimate normals.
#         rgb_output_name: Name of the RGB output.
#         depth_output_name: Name of the depth output.
#         use_bounding_box: Whether to use a bounding box to sample points.
#         bounding_box_min: Minimum of the bounding box.
#         bounding_box_max: Maximum of the bounding box.

#     Returns:
#         Point cloud.
#     """

#     progress = Progress(
#         TextColumn(":cloud: Computing Point Cloud :cloud:"),
#         BarColumn(),
#         TaskProgressColumn(show_speed=True),
#         TimeRemainingColumn(elapsed_when_finished=True, compact=True),
#     )
#     points = []
#     rgbs = []
#     with progress as progress_bar:
#         task = progress_bar.add_task("Generating Point Cloud", total=num_points)
#         while not progress_bar.finished:
#             with torch.no_grad():
#                 ray_bundle, _ = pipeline.datamanager.next_train(0)
#                 outputs = pipeline.model(ray_bundle)
#             if rgb_output_name not in outputs:
#                 CONSOLE.rule("Error", style="red")
#                 CONSOLE.print(f"Could not find {rgb_output_name} in the model outputs", justify="center")
#                 CONSOLE.print(f"Please set --rgb_output_name to one of: {outputs.keys()}", justify="center")
#                 sys.exit(1)
#             if depth_output_name not in outputs:
#                 CONSOLE.rule("Error", style="red")
#                 CONSOLE.print(f"Could not find {depth_output_name} in the model outputs", justify="center")
#                 CONSOLE.print(f"Please set --depth_output_name to one of: {outputs.keys()}", justify="center")
#                 sys.exit(1)
#             rgb = outputs[rgb_output_name]
#             depth = outputs[depth_output_name]
#             point = ray_bundle.origins + ray_bundle.directions * depth

#             if use_bounding_box:
#                 comp_l = torch.tensor(bounding_box_min, device=point.device)
#                 comp_m = torch.tensor(bounding_box_max, device=point.device)
#                 assert torch.all(
#                     comp_l < comp_m
#                 ), f"Bounding box min {bounding_box_min} must be smaller than max {bounding_box_max}"
#                 mask = torch.all(torch.concat([point > comp_l, point < comp_m], dim=-1), dim=-1)
#                 point = point[mask]
#                 rgb = rgb[mask]

#             points.append(point)
#             rgbs.append(rgb)
#             progress.advance(task, point.shape[0])
#     points = torch.cat(points, dim=0)
#     rgbs = torch.cat(rgbs, dim=0)

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points.float().cpu().numpy())
#     pcd.colors = o3d.utility.Vector3dVector(rgbs.float().cpu().numpy())

#     if remove_outliers:
#         CONSOLE.print("Cleaning Point Cloud")
#         pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
#         print("\033[A\033[A")
#         CONSOLE.print("[bold green]:white_check_mark: Cleaning Point Cloud")

#     if estimate_normals:
#         CONSOLE.print("Estimating Point Cloud Normals")
#         pcd.estimate_normals()
#         print("\033[A\033[A")
#         CONSOLE.print("[bold green]:white_check_mark: Estimating Point Cloud Normals")

#     return pcd


def generate_point_cloud(
    pipeline: Pipeline,
    num_points: int = 1000000,
    remove_outliers: bool = True,
    estimate_normals: bool = False,
    rgb_output_name: str = "rgb",
    depth_output_name: str = "depth",
    use_bounding_box: bool = True,
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> o3d.geometry.PointCloud:
    """Generate a point cloud from a nerf.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        remove_outliers: Whether to remove outliers.
        estimate_normals: Whether to estimate normals.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        use_bounding_box: Whether to use a bounding box to sample points.
        bounding_box_min: Minimum of the bounding box.
        bounding_box_max: Maximum of the bounding box.

    Returns:
        Point cloud.
    """

    progress = Progress(
        TextColumn(":cloud: Computing Point Cloud :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    points = []
    rgbs = []
    n_pts = 0
    while n_pts < num_points:
        with torch.no_grad():
            ray_bundle, _ = pipeline.datamanager.next_train(0)
            outputs = pipeline.model(ray_bundle)
        if rgb_output_name not in outputs:
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(f"Could not find {rgb_output_name} in the model outputs", justify="center")
            CONSOLE.print(f"Please set --rgb_output_name to one of: {outputs.keys()}", justify="center")
            sys.exit(1)
        if depth_output_name not in outputs:
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(f"Could not find {depth_output_name} in the model outputs", justify="center")
            CONSOLE.print(f"Please set --depth_output_name to one of: {outputs.keys()}", justify="center")
            sys.exit(1)
        rgb = outputs[rgb_output_name]
        depth = outputs[depth_output_name]
        point = ray_bundle.origins + ray_bundle.directions * depth

        if use_bounding_box:
            comp_l = torch.tensor(bounding_box_min, device=point.device)
            comp_m = torch.tensor(bounding_box_max, device=point.device)
            assert torch.all(
                comp_l < comp_m
            ), f"Bounding box min {bounding_box_min} must be smaller than max {bounding_box_max}"
            mask = torch.all(torch.concat([point > comp_l, point < comp_m], dim=-1), dim=-1)
            point = point[mask]
            rgb = rgb[mask]

        points.append(point)
        rgbs.append(rgb)
        n_pts += point.shape[0]

    points = torch.cat(points, dim=0)
    rgbs = torch.cat(rgbs, dim=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.float().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(rgbs.float().cpu().numpy())

    if remove_outliers:
        CONSOLE.print("Cleaning Point Cloud")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Cleaning Point Cloud")

    if estimate_normals:
        CONSOLE.print("Estimating Point Cloud Normals")
        pcd.estimate_normals()
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Estimating Point Cloud Normals")

    return pcd


def generate_point_cloud_allimages(
    pipeline: Pipeline,
    num_points: int = 1000000,
    remove_outliers: bool = True,
    estimate_normals: bool = False,
    rgb_output_name: str = "rgb",
    depth_output_name: str = "depth",
    use_bounding_box: bool = True,
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    outdir: str = "",
) -> None:
    """Generate a point cloud from a nerf.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        remove_outliers: Whether to remove outliers.
        estimate_normals: Whether to estimate normals.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        use_bounding_box: Whether to use a bounding box to sample points.
        bounding_box_min: Minimum of the bounding box.
        bounding_box_max: Maximum of the bounding box.

    Returns:
        Point cloud.
    """

    progress = Progress(
        TextColumn(":cloud: Computing Point Cloud :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )

    n_pts = 0
    pcd_cnt = 0
    iter_eval_image_dataloader = iter(pipeline.datamanager.fixed_indices_eval_dataloader)
    out_bbox_data = open(os.path.join(outdir, "bbox.txt"), "w", encoding="utf-8")
    for image_id in range(pipeline.datamanager.eval_dataset.__len__()):
        print(image_id)
        with torch.no_grad():
            ## reimplement pipeline.datamanager.next_eval(0)

            points = []
            rgbs = []
            pipeline.datamanager.eval_count += 1
            ray_bundle, image_dict = next(iter_eval_image_dataloader)
            ray_bundle = ray_bundle.flatten()
            batch_num = math.ceil(ray_bundle.shape[0] / pipeline.datamanager.train_pixel_sampler.num_rays_per_batch)
            for i in range(batch_num):
                ray_bundle_batch = ray_bundle[
                    i
                    * pipeline.datamanager.train_pixel_sampler.num_rays_per_batch : (i + 1)
                    * pipeline.datamanager.train_pixel_sampler.num_rays_per_batch
                ]
                outputs = pipeline.model(ray_bundle_batch)
                rgb = outputs[rgb_output_name]
                depth = outputs[depth_output_name]
                depth_max = torch.max(depth)
                if depth_max > 5:
                    a = 1
                depth = torch.clamp(depth, min=ray_bundle_batch.nears, max=ray_bundle_batch.fars)
                depth_max = torch.max(depth)
                point = ray_bundle_batch.origins + ray_bundle_batch.directions * depth
                points.append(point)
                rgbs.append(rgb)
            points = torch.cat(points, dim=0)
            rgbs = torch.cat(rgbs, dim=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.float().cpu().numpy())
            pcd.colors = o3d.utility.Vector3dVector(rgbs.float().cpu().numpy())

            if remove_outliers:
                pcd = pcd.voxel_down_sample(voxel_size=0.0002 * 5)
                # CONSOLE.print("Cleaning Point Cloud")
                # pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)
                # print("\033[A\033[A")
                # CONSOLE.print("[bold green]:white_check_mark: Cleaning Point Cloud")

            if estimate_normals:
                CONSOLE.print("Estimating Point Cloud Normals")
                pcd.estimate_normals()
                print("\033[A\033[A")
                CONSOLE.print("[bold green]:white_check_mark: Estimating Point Cloud Normals")
            bbox_max = pcd.get_max_bound()
            bbox_min = pcd.get_min_bound()
            out_bbox_data.write(
                "{} {} {} {} {} {} {}\n".format(
                    image_id, bbox_min[0], bbox_min[1], bbox_min[2], bbox_max[0], bbox_max[1], bbox_max[2]
                )
            )
            pcd_path = os.path.join(outdir, str(image_id) + ".ply")
            o3d.io.write_point_cloud(pcd_path, pcd)
    out_bbox_data.close()

    return None

def generate_all_images(
    pipeline: Pipeline,
    num_points: int = 1000000,
    remove_outliers: bool = True,
    estimate_normals: bool = False,
    rgb_output_name: str = "rgb",
    depth_output_name: str = "depth",
    use_bounding_box: bool = True,
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    outdir: str = "",
) -> None:
    """Generate a point cloud from a nerf.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        remove_outliers: Whether to remove outliers.
        estimate_normals: Whether to estimate normals.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        use_bounding_box: Whether to use a bounding box to sample points.
        bounding_box_min: Minimum of the bounding box.
        bounding_box_max: Maximum of the bounding box.

    Returns:
        Point cloud.
    """
    import cv2
    import numpy as np
    progress = Progress(
        TextColumn(":cloud: Computing Point Cloud :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )

    iter_eval_image_dataloader = iter(pipeline.datamanager.fixed_indices_eval_dataloader)
    out_data = open(os.path.join(outdir, "eval.txt"), "w", encoding="utf-8")
    from torchmetrics import PeakSignalNoiseRatio
    from torchmetrics.functional import structural_similarity_index_measure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    psnr = PeakSignalNoiseRatio(data_range=1.0)
    ssim = structural_similarity_index_measure
    lpips = LearnedPerceptualImagePatchSimilarity()
    num_imgs=pipeline.datamanager.eval_dataset.__len__()
    num_imgs=50
    for image_id in range(num_imgs):
        print(image_id)
        with torch.no_grad():
            rgbs = []
            pipeline.datamanager.eval_count += 1
            ray_bundle, image_dict = next(iter_eval_image_dataloader)
            height=ray_bundle.shape[0]
            width=ray_bundle.shape[1]
            ray_bundle = ray_bundle.flatten()
            batch_num = math.ceil(ray_bundle.shape[0] / pipeline.datamanager.train_pixel_sampler.num_rays_per_batch)
            for i in range(batch_num):
                ray_bundle_batch = ray_bundle[
                    i
                    * pipeline.datamanager.train_pixel_sampler.num_rays_per_batch : (i + 1)
                    * pipeline.datamanager.train_pixel_sampler.num_rays_per_batch
                ]
                outputs = pipeline.model(ray_bundle_batch)
                rgb = outputs[rgb_output_name]
                depth = outputs[depth_output_name]
                depth_max = torch.max(depth)
                depth = torch.clamp(depth, min=ray_bundle_batch.nears, max=ray_bundle_batch.fars)
                depth_max = torch.max(depth)
                rgbs.append(rgb)
            rgbs = torch.cat(rgbs, dim=0)
            rgb_arr=rgbs.view(height,width,3).cpu()
            pred = torch.moveaxis(rgb_arr, -1, 0)[None, ...]
            gt = torch.moveaxis(image_dict['image'], -1, 0)[None, ...]
            psnr1=psnr(pred,gt)
            ssim1=ssim(pred,gt)
            lpips1=lpips(pred,gt)

            rgb_arr=(rgb_arr.float().numpy()*255).astype(np.uint8)
            b=np.expand_dims(rgb_arr[:,:,0],axis=2)
            g=np.expand_dims(rgb_arr[:,:,1],axis=2)
            r=np.expand_dims(rgb_arr[:,:,2],axis=2)

            cv2.imwrite(os.path.join(outdir,str(image_id)+'.jpg'),np.concatenate([r,g,b],axis=2))
            out_data.write("{} {} {}\n".format(psnr1.float(),ssim1.float(),lpips1.float()))
    out_data.close()
            
            

    return None
