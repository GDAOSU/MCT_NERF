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

"""
Export utils such as structs, point cloud generation, and rendering code.
"""

# pylint: disable=no-member

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import pymeshlab
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torchtyping import TensorType

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.rich_utils import ItersPerSecColumn

CONSOLE = Console(width=120)


@dataclass
class Mesh:
    """Class for a mesh."""

    vertices: TensorType["num_verts", 3]
    """Vertices of the mesh."""
    faces: TensorType["num_faces", 3]
    """Faces of the mesh."""
    normals: TensorType["num_verts", 3]
    """Normals of the mesh."""
    colors: Optional[TensorType["num_verts", 3]] = None
    """Colors of the mesh."""


def get_mesh_from_pymeshlab_mesh(mesh: pymeshlab.Mesh) -> Mesh:
    """Get a Mesh from a pymeshlab mesh.
    See https://pymeshlab.readthedocs.io/en/0.1.5/classes/mesh.html for details.
    """
    return Mesh(
        vertices=torch.from_numpy(mesh.vertex_matrix()).float(),
        faces=torch.from_numpy(mesh.face_matrix()).long(),
        normals=torch.from_numpy(np.copy(mesh.vertex_normal_matrix())).float(),
        colors=torch.from_numpy(mesh.vertex_color_matrix()).float(),
    )


def get_mesh_from_filename(filename: str, target_num_faces: Optional[int] = None) -> Mesh:
    """Get a Mesh from a filename."""
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(filename)
    if target_num_faces is not None:
        CONSOLE.print("Running meshing decimation with quadric edge collapse")
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_num_faces)
    mesh = ms.current_mesh()
    return get_mesh_from_pymeshlab_mesh(mesh)


def generate_depth(
    pipeline: Pipeline,
    rgb_output_name: str = "rgb_fine",
    depth_output_name: str = "depth_fine",
    output_dir: str="",
) -> o3d.geometry.PointCloud:
    """Generate a point cloud from a nerf.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        remove_outliers: Whether to remove outliers.
        estimate_normals: Whether to estimate normals.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        normal_output_name: Name of the normal output.
        use_bounding_box: Whether to use a bounding box to sample points.
        bounding_box_min: Minimum of the bounding box.
        bounding_box_max: Maximum of the bounding box.
        std_ratio: Threshold based on STD of the average distances across the point cloud to remove outliers.

    Returns:
        Point cloud.
    """

    # pylint: disable=too-many-statements
    points = []
    rgbs = []
    normals = []
    num_images = len(pipeline.datamanager.fixed_indices_eval_dataloader)
    import os

    import cv2
    import numpy as np
    from PIL import Image
    with torch.no_grad():
        cnt=0
        for camera_ray_bundle, batch in pipeline.datamanager.fixed_indices_eval_dataloader:
            # time this the following line
            height, width = camera_ray_bundle.shape
            num_rays = height * width
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

            gt=batch['image'].detach().cpu().numpy()*255
            gt.astype(np.uint8)
            gt=cv2.cvtColor(gt,cv2.COLOR_BGR2RGB)
            rgb = outputs[rgb_output_name].detach().cpu().numpy()*255
            rgb.astype(np.uint8)
            rgb=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
            depth = outputs[depth_output_name].squeeze().detach().cpu().numpy()
            
            im = Image.fromarray(depth)
            im.save(os.path.join(output_dir,str(cnt)+'_depth.tif'))
            cv2.imwrite(os.path.join(output_dir,str(cnt)+'_rgb.png'),rgb)
            cv2.imwrite(os.path.join(output_dir,str(cnt)+'_gt.png'),gt)
            cnt+=1


    return depth

def generate_point_cloud_all(
    pipeline: Pipeline,
    rgb_output_name: str = "rgb_fine",
    depth_output_name: str = "depth_fine",
    output_dir: str="",
    shiftx: float=0.0,
    shifty: float=0.0,
    shiftz: float=0.0,
    scale:float=0.0

) -> o3d.geometry.PointCloud:
    """Generate a point cloud from a nerf.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        remove_outliers: Whether to remove outliers.
        estimate_normals: Whether to estimate normals.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        normal_output_name: Name of the normal output.
        use_bounding_box: Whether to use a bounding box to sample points.
        bounding_box_min: Minimum of the bounding box.
        bounding_box_max: Maximum of the bounding box.
        std_ratio: Threshold based on STD of the average distances across the point cloud to remove outliers.

    Returns:
        Point cloud.
    """

    # pylint: disable=too-many-statements
    points = []
    rgbs = []
    normals = []
    num_images = len(pipeline.datamanager.fixed_indices_eval_dataloader)
    import os

    import cv2
    import numpy as np
    from PIL import Image
    with torch.no_grad():
        cnt=0
        for camera_ray_bundle, batch in pipeline.datamanager.fixed_indices_eval_dataloader:
            # time this the following line
            height, width = camera_ray_bundle.shape
            num_rays = height * width
            num_rays_per_chunk = pipeline.model.config.eval_num_rays_per_chunk
            #split big image into batches of 500000 pixels
            for i in range(0, num_rays, 500000):
                i_next=min(num_rays,i+500000)
                pt_name=str(cnt)+"_"+str(i)+".ply"
                pt_path=str(output_dir / pt_name)
                if os.path.exists(pt_path):
                    continue

                outputs_lists = defaultdict(list)
                for j in range(i,i_next,num_rays_per_chunk):
                    start_idx = j
                    end_idx = j + num_rays_per_chunk
                    ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                    outputs = pipeline.model(ray_bundle=ray_bundle)
                    point=ray_bundle.origins+ray_bundle.directions*outputs[depth_output_name]
                    rgb=outputs[rgb_output_name]
                    point=point.view(-1,3)
                    rgb=rgb.view(-1,3)
                    outputs_lists["points"].append(point)
                    outputs_lists["rgb"].append(rgb)
                points=torch.cat(outputs_lists["points"])
                rgbs=torch.cat(outputs_lists["rgb"])
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points.double().cpu().numpy())
                pcd.colors = o3d.utility.Vector3dVector(rgbs.double().cpu().numpy())
                
                shift=np.array([shiftx,shifty,shiftz])
                pcd.scale(scale,center=np.zeros(3))
                pcd.translate(shift)

                tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
                # The legacy PLY writer converts colors to UInt8,
                # let us do the same to save space.
                tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)

                o3d.t.io.write_point_cloud(str(output_dir / pt_name), tpcd)
            # outputs_lists = defaultdict(list)
            # outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

            # rgb = outputs[rgb_output_name]
            # depth = outputs[depth_output_name]
            # point = camera_ray_bundle.origins + camera_ray_bundle.directions * depth
            # point=point.view(-1,3)
            # rgb=rgb.view(-1,3)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(point.double().cpu().numpy())
            # pcd.colors = o3d.utility.Vector3dVector(rgb.double().cpu().numpy())

            # shift=np.array([shiftx,shifty,shiftz])
            # pcd.scale(scale,center=np.zeros(3))
            # pcd.translate(shift)
            
            # tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
            # # The legacy PLY writer converts colors to UInt8,
            # # let us do the same to save space.
            # tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)
            # pt_name=str(cnt)+".ply"
            # o3d.t.io.write_point_cloud(str(output_dir / pt_name), tpcd)
            cnt+=1


    #return depth
    return 1



def generate_point_cloud(
    pipeline: Pipeline,
    num_points: int = 1000000,
    remove_outliers: bool = True,
    estimate_normals: bool = False,
    rgb_output_name: str = "rgb_fine",
    depth_output_name: str = "depth_fine",
    normal_output_name: Optional[str] = None,
    use_bounding_box: bool = True,
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    std_ratio: float = 10.0,
) -> o3d.geometry.PointCloud:
    """Generate a point cloud from a nerf.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        remove_outliers: Whether to remove outliers.
        estimate_normals: Whether to estimate normals.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        normal_output_name: Name of the normal output.
        use_bounding_box: Whether to use a bounding box to sample points.
        bounding_box_min: Minimum of the bounding box.
        bounding_box_max: Maximum of the bounding box.
        std_ratio: Threshold based on STD of the average distances across the point cloud to remove outliers.

    Returns:
        Point cloud.
    """

    # pylint: disable=too-many-statements

    progress = Progress(
        TextColumn(":cloud: Computing Point Cloud :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    points = []
    rgbs = []
    normals = []
    with progress as progress_bar:
        task = progress_bar.add_task("Generating Point Cloud", total=num_points)
        while not progress_bar.finished:
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
            if normal_output_name is not None:
                if normal_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {normal_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --normal_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)
                normal = outputs[normal_output_name]
                assert (
                    torch.min(normal) >= 0.0 and torch.max(normal) <= 1.0
                ), "Normal values from method output must be in [0, 1]"
                normal = (normal * 2.0) - 1.0
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
                if normal_output_name is not None:
                    normal = normal[mask]

            points.append(point)
            rgbs.append(rgb)
            if normal_output_name is not None:
                normals.append(normal)
            progress.advance(task, point.shape[0])
    points = torch.cat(points, dim=0)
    rgbs = torch.cat(rgbs, dim=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.double().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(rgbs.double().cpu().numpy())

    ind = None
    if remove_outliers:
        CONSOLE.print("Cleaning Point Cloud")
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Cleaning Point Cloud")

    # either estimate_normals or normal_output_name, not both
    if estimate_normals:
        if normal_output_name is not None:
            CONSOLE.rule("Error", style="red")
            CONSOLE.print("Cannot estimate normals and use normal_output_name at the same time", justify="center")
            sys.exit(1)
        CONSOLE.print("Estimating Point Cloud Normals")
        pcd.estimate_normals()
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Estimating Point Cloud Normals")
    elif normal_output_name is not None:
        normals = torch.cat(normals, dim=0)
        if ind is not None:
            # mask out normals for points that were removed with remove_outliers
            normals = normals[ind]
        pcd.normals = o3d.utility.Vector3dVector(normals.double().cpu().numpy())

    return pcd


def render_trajectory(
    pipeline: Pipeline,
    cameras: Cameras,
    rgb_output_name: str,
    depth_output_name: str,
    rendered_resolution_scaling_factor: float = 1.0,
    disable_distortion: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Helper function to create a video of a trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        disable_distortion: Whether to disable distortion.

    Returns:
        List of rgb images, list of depth images.
    """
    images = []
    depths = []
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)

    progress = Progress(
        TextColumn(":cloud: Computing rgb and depth images :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    with progress:
        for camera_idx in progress.track(range(cameras.size), description=""):
            camera_ray_bundle = cameras.generate_rays(
                camera_indices=camera_idx, disable_distortion=disable_distortion
            ).to(pipeline.device)
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
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
            images.append(outputs[rgb_output_name].cpu().numpy())
            depths.append(outputs[depth_output_name].cpu().numpy())
    return images, depths

def generate_point_cloud_all_mct(
    pipeline: Pipeline,
    rgb_output_name: str = "rgb_fine",
    depth_output_name: str = "depth_fine",
    output_dir: str="",
    num_pts: int=1000000,
    shiftx: float=0.0,
    shifty: float=0.0,
    shiftz: float=0.0,
    scale:float=0.0

) -> o3d.geometry.PointCloud:
    """Generate a point cloud from a nerf.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        remove_outliers: Whether to remove outliers.
        estimate_normals: Whether to estimate normals.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        normal_output_name: Name of the normal output.
        use_bounding_box: Whether to use a bounding box to sample points.
        bounding_box_min: Minimum of the bounding box.
        bounding_box_max: Maximum of the bounding box.
        std_ratio: Threshold based on STD of the average distances across the point cloud to remove outliers.

    Returns:
        Point cloud.
    """

    # pylint: disable=too-many-statements
    import os

    import cv2
    import numpy as np
    from PIL import Image
    num_imgs=len(pipeline.datamanager.fixed_indices_eval_dataloader)
    num_pts_per_img=int(num_pts/num_imgs)
    skip_image=2
    with torch.no_grad():
        cnt=0
        points=[]
        rgbs=[]
        for camera_ray_bundle, batch in pipeline.datamanager.fixed_indices_eval_dataloader:
            if batch["image_idx"]%skip_image!=0:
                continue
            # time this the following line
            camera_ray_bundle=camera_ray_bundle.to(torch.device("cpu"))
            height, width = camera_ray_bundle.shape
            pt_name=str(cnt)+"_.ply"
            pt_path=str(output_dir / pt_name)
            if os.path.exists(pt_path):
                continue

            num_rays_per_chunk = pipeline.model.config.eval_num_rays_per_chunk
            image_height, image_width = camera_ray_bundle.origins.shape[:2]
            num_rays = len(camera_ray_bundle)
            depth_list = []
            rgb_list=[]
            for i in range(0, num_rays, num_rays_per_chunk):
                start_idx = i
                end_idx = i + num_rays_per_chunk
                ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                outputs = pipeline.model.forward(ray_bundle=ray_bundle.to(pipeline.device))
                depth_list.append(outputs['depth_fine'].cpu())
                rgb_list.append(outputs['rgb_fine'].cpu())

            depth=torch.cat(depth_list).view(image_height, image_width, -1)
            rgb=torch.cat(rgb_list).view(image_height, image_width, -1)

            point=camera_ray_bundle.origins+camera_ray_bundle.directions*depth
            point=point.view(-1,3)
            rgb=rgb.view(-1,3)

            perm=torch.randperm(height*width)
            indices=perm[:num_pts_per_img]
            point=point[indices,:]
            rgb=rgb[indices,:]
            points.append(point)
            rgbs.append(rgb)
        points=torch.cat(points)
        rgbs=torch.cat(rgbs)
        return points,rgbs
        points=torch.cat(points)
        rgbs=torch.cat(rgbs)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.double().numpy())
        pcd.colors = o3d.utility.Vector3dVector(rgbs.double().numpy())
        
        shift=np.array([shiftx,shifty,shiftz])
        pcd.scale(scale,center=np.zeros(3))
        pcd.translate(shift)

        tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        # The legacy PLY writer converts colors to UInt8,
        # let us do the same to save space.
        tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)

        #o3d.t.io.write_point_cloud(str(output_dir / pt_name), tpcd)
        return tpcd




def collect_camera_poses_for_dataset(dataset: Optional[InputDataset]) -> List[Dict[str, Any]]:
    """Collects rescaled, translated and optimised camera poses for a dataset.

    Args:
        dataset: Dataset to collect camera poses for.

    Returns:
        List of dicts containing camera poses.
    """

    if dataset is None:
        return []

    cameras = dataset.cameras
    image_filenames = dataset.image_filenames

    frames: List[Dict[str, Any]] = []

    # new cameras are in cameras, whereas image paths are stored in a private member of the dataset
    for idx in range(len(cameras)):
        image_filename = image_filenames[idx]
        transform = cameras.camera_to_worlds[idx].tolist()
        frames.append(
            {
                "file_path": str(image_filename),
                "transform": transform,
            }
        )

    return frames


def collect_camera_poses(pipeline: VanillaPipeline) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collects camera poses for train and eval datasets.

    Args:
        pipeline: Pipeline to evaluate with.

    Returns:
        List of train camera poses, list of eval camera poses.
    """

    train_dataset = pipeline.datamanager.train_dataset
    assert isinstance(train_dataset, InputDataset)

    eval_dataset = pipeline.datamanager.eval_dataset
    assert isinstance(eval_dataset, InputDataset)

    train_frames = collect_camera_poses_for_dataset(train_dataset)
    eval_frames = collect_camera_poses_for_dataset(eval_dataset)

    return train_frames, eval_frames
