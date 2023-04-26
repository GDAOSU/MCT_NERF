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
Camera transformation helper code.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
from torchtyping import TensorType
from typing_extensions import Literal

_EPS = np.finfo(float).eps * 4.0


def unit_vector(data, axis: Optional[int] = None) -> np.ndarray:
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    Args:
        axis: the axis along which to normalize into unit vector
        out: where to write out the data to. If None, returns a new np ndarray
    """
    data = np.array(data, dtype=np.float64, copy=True)
    if data.ndim == 1:
        data /= math.sqrt(np.dot(data, data))
        return data
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    return data


def quaternion_from_matrix(matrix, isprecise: bool = False) -> np.ndarray:
    """Return quaternion from rotation matrix.

    Args:
        matrix: rotation matrix to obtain quaternion
        isprecise: if True, input matrix is assumed to be precise rotation matrix and a faster algorithm is used.
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, 0.0, 0.0, 0.0],
                [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[np.array([3, 0, 1, 2]), np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def quaternion_slerp(quat0, quat1, fraction: float, spin: int = 0, shortestpath: bool = True) -> np.ndarray:
    """Return spherical linear interpolation between two quaternions.
    Args:
        quat0: first quaternion
        quat1: second quaternion
        fraction: how much to interpolate between quat0 vs quat1 (if 0, closer to quat0; if 1, closer to quat1)
        spin: how much of an additional spin to place on the interpolation
        shortestpath: whether to return the short or long path to rotation
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if q0 is None or q1 is None:
        raise ValueError("Input quaternions invalid.")
    if fraction == 0.0:
        return q0
    if fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        np.negative(q1, q1)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def quaternion_matrix(quaternion) -> np.ndarray:
    """Return homogeneous rotation matrix from quaternion.

    Args:
        quaternion: value to convert to matrix
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def get_interpolated_poses(pose_a, pose_b, steps: int = 10) -> List[float]:
    """Return interpolation of poses with specified number of steps.
    Args:
        poseA: first pose
        poseB: second pose
        steps: number of steps the interpolated pose path should contain
    """

    quat_a = quaternion_from_matrix(pose_a[:3, :3])
    quat_b = quaternion_from_matrix(pose_b[:3, :3])

    ts = np.linspace(0, 1, steps)
    quats = [quaternion_slerp(quat_a, quat_b, t) for t in ts]
    trans = [(1 - t) * pose_a[:3, 3] + t * pose_b[:3, 3] for t in ts]

    poses_ab = []
    for quat, tran in zip(quats, trans):
        pose = np.identity(4)
        pose[:3, :3] = quaternion_matrix(quat)[:3, :3]
        pose[:3, 3] = tran
        poses_ab.append(pose)
    return poses_ab


def get_interpolated_k(k_a, k_b, steps: int = 10) -> TensorType[3, 4]:
    """
    Returns interpolated path between two camera poses with specified number of steps.

    Args:
        KA: camera matrix 1
        KB: camera matrix 2
        steps: number of steps the interpolated pose path should contain
    """
    Ks = []
    ts = np.linspace(0, 1, steps)
    for t in ts:
        new_k = k_a * (1.0 - t) + k_b * t
        Ks.append(new_k)
    return Ks


def get_interpolated_poses_many(
    poses: TensorType["num_poses", 3, 4],
    Ks: TensorType["num_poses", 3, 3],
    steps_per_transition=10,
) -> Tuple[TensorType["num_poses", 3, 4], TensorType["num_poses", 3, 3]]:
    """Return interpolated poses for many camera poses.

    Args:
        poses: list of camera poses
        Ks: list of camera intrinsics
        steps_per_transition: number of steps per transition

    Returns:
        tuple of new poses and intrinsics
    """
    traj = []
    Ks = []
    for idx in range(poses.shape[0] - 1):
        pose_a = poses[idx]
        pose_b = poses[idx + 1]
        poses_ab = get_interpolated_poses(pose_a, pose_b, steps=steps_per_transition)
        traj += poses_ab
        Ks += get_interpolated_k(Ks[idx], Ks[idx + 1], steps_per_transition)
    return torch.stack(traj, dim=0), torch.stack(Ks, dim=0)


def normalize(x) -> TensorType[...]:
    """Returns a normalized vector."""
    return x / torch.linalg.norm(x)


def viewmatrix(lookat, up, pos) -> TensorType[...]:
    """Returns a camera transformation matrix.

    Args:
        lookat: The direction the camera is looking.
        up: The upward direction of the camera.
        pos: The position of the camera.

    Returns:
        A camera transformation matrix.
    """
    vec2 = normalize(lookat)
    vec1_avg = normalize(up)
    vec0 = normalize(torch.cross(vec1_avg, vec2))
    vec1 = normalize(torch.cross(vec2, vec0))
    m = torch.stack([vec0, vec1, vec2, pos], 1)
    return m


def get_distortion_params(
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    k4: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> TensorType[...]:
    """Returns a distortion parameters matrix.

    Args:
        k1: The first radial distortion parameter.
        k2: The second radial distortion parameter.
        k3: The third radial distortion parameter.
        k4: The fourth radial distortion parameter.
        p1: The first tangential distortion parameter.
        p2: The second tangential distortion parameter.
    Returns:
        torch.Tensor: A distortion parameters matrix.
    """
    return torch.Tensor([k1, k2, k3, k4, p1, p2])


@torch.jit.script
def _compute_residual_and_jacobian(
    x: torch.Tensor,
    y: torch.Tensor,
    xd: torch.Tensor,
    yd: torch.Tensor,
    distortion_params: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]:
    """Auxiliary function of radial_and_tangential_undistort() that computes residuals and jacobians.
    Adapted from MultiNeRF:
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L427-L474

    Args:
        x: The updated x coordinates.
        y: The updated y coordinates.
        xd: The distorted x coordinates.
        yd: The distorted y coordinates.
        distortion_params: The distortion parameters [k1, k2, k3, k4, p1, p2].

    Returns:
        The residuals (fx, fy) and jacobians (fx_x, fx_y, fy_x, fy_y).
    """

    k1 = distortion_params[..., 0]
    k2 = distortion_params[..., 1]
    k3 = distortion_params[..., 2]
    k4 = distortion_params[..., 3]
    p1 = distortion_params[..., 4]
    p2 = distortion_params[..., 5]

    # let r(x, y) = x^2 + y^2;
    #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3 +
    #                   k4 * r(x, y)^4;
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of d over [x, y]
    d_r = k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4))
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y


@torch.jit.script
def radial_and_tangential_undistort(
    coords: torch.Tensor,
    distortion_params: torch.Tensor,
    eps: float = 1e-3,
    max_iterations: int = 10,
) -> torch.Tensor:
    """Computes undistorted coords given opencv distortion parameters.
    Addapted from MultiNeRF
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L477-L509

    Args:
        coords: The distorted coordinates.
        distortion_params: The distortion parameters [k1, k2, k3, k4, p1, p2].
        eps: The epsilon for the convergence.
        max_iterations: The maximum number of iterations to perform.

    Returns:
        The undistorted coordinates.
    """

    # Initialize from the distorted point.
    x = coords[..., 0]
    y = coords[..., 1]

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=coords[..., 0], yd=coords[..., 1], distortion_params=distortion_params
        )
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = torch.where(torch.abs(denominator) > eps, x_numerator / denominator, torch.zeros_like(denominator))
        step_y = torch.where(torch.abs(denominator) > eps, y_numerator / denominator, torch.zeros_like(denominator))

        x = x + step_x
        y = y + step_y

    return torch.stack([x, y], dim=-1)


def rotation_matrix(a: TensorType[3], b: TensorType[3]) -> TensorType[3, 3]:
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (torch.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = torch.linalg.norm(v)
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return torch.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s**2 + 1e-8))


def auto_orient_and_center_poses(
    poses: TensorType["num_poses":..., 4, 4], method: Literal["pca", "up", "none"] = "up", center_poses: bool = True
) -> TensorType["num_poses":..., 3, 4]:
    """Orients and centers the poses. We provide two methods for orientation: pca and up.

    pca: Orient the poses so that the principal component of the points is aligned with the axes.
        This method works well when all of the cameras are in the same plane.
    up: Orient the poses so that the average up vector is aligned with the z axis.
        This method works well when images are not at arbitrary angles.


    Args:
        poses: The poses to orient.
        method: The method to use for orientation.
        center_poses: If True, the poses are centered around the origin.

    Returns:
        The oriented poses.
    """

    translation = poses[..., :3, 3]

    mean_translation = torch.mean(translation, dim=0)
    translation_diff = translation - mean_translation

    if center_poses:
        translation = mean_translation
    else:
        translation = torch.zeros_like(mean_translation)

    if method == "pca":
        _, eigvec = torch.linalg.eigh(translation_diff.T @ translation_diff)
        eigvec = torch.flip(eigvec, dims=(-1,))

        if torch.linalg.det(eigvec) < 0:
            eigvec[:, 2] = -eigvec[:, 2]

        transform = torch.cat([eigvec, eigvec @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses

        if oriented_poses.mean(axis=0)[2, 1] < 0:
            oriented_poses[:, 1:3] = -1 * oriented_poses[:, 1:3]
    elif method == "up":
        up = torch.mean(poses[:, :3, 1], dim=0)
        up = up / torch.linalg.norm(up)

        rotation = rotation_matrix(up, torch.Tensor([0, 0, 1]))
        transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses
    elif method == "none":
        oriented_poses = poses
        poses[:, :3, 3] -= translation

    return oriented_poses


def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, 2, device=c2w.device), torch.linspace(0, H - 1, 2, device=c2w.device)
    )  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()

    i, j = i + 0.5, j + 0.5

    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, mode="center"):
    rays_o, rays_d = get_rays(H, W, K, c2w)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    return rays_o, rays_d, viewdirs


def _compute_bbox_by_cam_frustrm_bounded(heights, widths, Ks, poses, nears, fars):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for H, W, K, c2w, near, far in zip(heights, widths, Ks, poses, nears, fars):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(H=H, W=W, K=K, c2w=c2w)
        pts_nf = torch.stack([rays_o + viewdirs * near, rays_o + viewdirs * far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0, 1, 2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0, 1, 2)))
    return xyz_min, xyz_max


def center_scale_poses_and_compute_frustum(poses, bbox=None):
    transform=torch.eye(4)

    scale_factor = 1
    use_bbox = False
    if bbox is not None:
        if bbox.shape[0] == 6:
            use_bbox = True

    if use_bbox:
        sampling_xyz_min = torch.tensor(bbox[:3])
        sampling_xyz_max = torch.tensor(bbox[3:])
        Cs = poses[:, :3, 3]
        Cs_min = Cs.amin((0))
        Cs_max = Cs.amax((0))
        xyz_min = torch.minimum(sampling_xyz_min, Cs_min)
        xyz_max = torch.maximum(sampling_xyz_max, Cs_max)
        translation = torch.tensor(
            [
                (sampling_xyz_min[0] + sampling_xyz_max[0]) / 2,
                (sampling_xyz_min[1] + sampling_xyz_max[1]) / 2,
                (sampling_xyz_min[2] + sampling_xyz_max[2]) / 2,
            ]
        )
        ##center
        poses[:, :3, 3] -= translation
        sampling_xyz_min -= translation
        sampling_xyz_max -= translation
        xyz_min -= translation
        xyz_max -= translation
        ## find the radius
        r1 = torch.norm(torch.tensor([sampling_xyz_min[0], sampling_xyz_min[1], sampling_xyz_min[2]]))
        r2 = torch.norm(torch.tensor([sampling_xyz_min[0], sampling_xyz_min[1], sampling_xyz_max[2]]))
        r3 = torch.norm(torch.tensor([sampling_xyz_min[0], sampling_xyz_max[1], sampling_xyz_min[2]]))
        r4 = torch.norm(torch.tensor([sampling_xyz_min[0], sampling_xyz_max[1], sampling_xyz_max[2]]))
        r5 = torch.norm(torch.tensor([sampling_xyz_max[0], sampling_xyz_min[1], sampling_xyz_min[2]]))
        r6 = torch.norm(torch.tensor([sampling_xyz_max[0], sampling_xyz_min[1], sampling_xyz_max[2]]))
        r7 = torch.norm(torch.tensor([sampling_xyz_max[0], sampling_xyz_max[1], sampling_xyz_min[2]]))
        r8 = torch.norm(torch.tensor([sampling_xyz_max[0], sampling_xyz_max[1], sampling_xyz_max[2]]))
        radius = torch.max(torch.tensor([r1, r2, r3, r4, r5, r6, r7, r8]))

        # scale = scale_factor / max(max(xyz_max[0] - xyz_min[0], xyz_max[1] - xyz_min[1]), xyz_max[2] - xyz_min[2])
        scale = scale_factor / radius
        poses[:, :3, 3] *= scale
        sampling_xyz_min *= scale
        sampling_xyz_max *= scale
        xyz_min *= scale
        xyz_max *= scale


        #compute transoform of poses from input space to nerf space
        transform[0,3]=translation[0]
        transform[1,3]=translation[1]
        transform[2,3]=translation[2]
        transform[0,0]=scale
        transform[1,1]=scale
        transform[2,2]=scale
        # bbox = np.array(bbox).astype(np.float64)
        # # center
        # translation = poses[..., :3, 3]
        # mean_translation = torch.mean(translation, dim=0)
        # translation = mean_translation
        # poses[:, :3, 3] -= translation
        # bbox[0] -= translation[0]
        # bbox[3] -= translation[0]
        # bbox[1] -= translation[1]
        # bbox[4] -= translation[1]
        # bbox[2] -= translation[2]
        # bbox[5] -= translation[2]

        # # scale by minimal near plane
        # sc = scale_factor / nears.min()
        # poses[:, :3, 3] *= sc
        # bbox *= sc.numpy()
        # sampling_xyz_min = torch.tensor(bbox[:3])
        # sampling_xyz_max = torch.tensor(bbox[3:])
        # Cs = poses[:, :3, 3]
        # Cs_min = Cs.amin((0))
        # Cs_max = Cs.amax((0))
        # xyz_min = torch.minimum(sampling_xyz_min, Cs_min)
        # xyz_max = torch.maximum(sampling_xyz_max, Cs_max)
        return xyz_min, xyz_max, sampling_xyz_min, sampling_xyz_max, poses, -translation,scale
    else:
        assert (use_bbox == False, "please give the bbox")
        # center
        # translation = poses[..., :3, 3]
        # mean_translation = torch.mean(translation, dim=0)
        # translation = mean_translation
        # poses[:, :3, 3] -= translation
        # # scale by minimal near plane
        # sc = scale_factor / nears.min()
        # poses[:, :3, 3] *= sc
        # nears *= sc
        # fars *= sc
        # ##compute frustum
        # sampling_xyz_min, sampling_xyz_max = _compute_bbox_by_cam_frustrm_bounded(
        #     heights, widths, Ks, poses, nears, fars
        # )
        # Cs = poses[:, :3, 3]
        # Cs_min = Cs.amin((0))
        # Cs_max = Cs.amax((0))
        # xyz_min = torch.minimum(sampling_xyz_min, Cs_min)
        # xyz_max = torch.maximum(sampling_xyz_max, Cs_max)
        # return xyz_min, xyz_max, sampling_xyz_min, sampling_xyz_max, poses, nears, fars


def get_rays(H, W, K, c2w, inverse_y=False, flip_x=False, flip_y=False, mode="center"):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=c2w.device), torch.linspace(0, H - 1, H, device=c2w.device)
    )  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == "lefttop":
        pass
    elif mode == "center":
        i, j = i + 0.5, j + 0.5
    elif mode == "random":
        i = i + torch.rand_like(i)
        j = j + torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i - K[0][2]) / K[0][0], (j - K[1][2]) / K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def filter_images_aoi_aabb(poses, heights, widths, fxs, fys, cxs, cys, aoi_aabb):
    from nerfstudio.data.scene_box import SceneBox
    from nerfstudio.model_components.scene_colliders import AABBBoxCollider

    aoi_aabb = torch.tensor([aoi_aabb[:3], aoi_aabb[3:]])
    scene_box = SceneBox(aoi_aabb)
    collider = AABBBoxCollider(scene_box, near_plane=0, far_plane=10000)

    valid_ids = []
    for i in range(poses.shape[0]):
        pose = poses[i, :, :]
        height = heights[i]
        width = widths[i]
        fx = fxs[i]
        fy = fys[i]
        cx = cxs[i]
        cy = cys[i]
        K = np.zeros((3, 3))
        K[0, 0] = fx
        K[0, 2] = cx
        K[1, 1] = fy
        K[1, 2] = cy
        rays_o, rays_d = get_rays(height, width, K, pose)
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)
        select_ind = torch.range(0, rays_o.shape[0] - 1, 100).type(torch.int64)
        rays_o = rays_o[select_ind, :]
        rays_d = rays_d[select_ind, :]
        nears, fars = collider._intersect_with_aabb(rays_o, rays_d, scene_box.aabb)
        diff = fars - nears
        num_valid = torch.sum(diff > 1e-5)
        valid_ratio = float(num_valid / (rays_o.shape[0]))
        if valid_ratio > 0.3:
            valid_ids.append(i)
    return valid_ids
