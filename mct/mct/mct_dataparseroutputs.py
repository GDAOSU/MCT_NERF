
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

from mct import mct_camera_utils
from mct.mct_scene_box import SceneBox
from mct.mct_utils import read_cameras_text, read_images_text
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
    transform_poses_to_original_space,
)


@dataclass
class MCTDataParserOutputs(DataparserOutputs):
    """Dataparser outputs for the which will be used by the DataManager
    for creating RayBundle and RayGT objects."""

    image_filenames: List[Path]
    """Filenames for the images."""
    cameras: Cameras
    sampling_box: SceneBox=SceneBox()
    """Camera object storing collection of camera information in dataset."""
    alpha_color: Optional[TensorType[3]] = None
    """Color of dataset background."""
    scene_box: SceneBox = SceneBox()
    """Scene box of dataset. Used to bound the scene or provide the scene scale depending on model."""
    mask_filenames: Optional[List[Path]] = None
    """Filenames for any masks that are required"""
    metadata: Dict[str, Any] = to_immutable_dict({})
    """Dictionary of any metadata that be required for the given experiment.
    Will be processed by the InputDataset to create any additional tensors that may be required.
    """
    dataparser_transform: TensorType[3, 4] = torch.eye(4)[:3, :]
    """Transform applied by the dataparser."""
    dataparser_scale: float = 1.0
    """Scale applied by the dataparser."""
    dataparser_shift: TensorType[3] = torch.zeros(0)
    """Scale applied by the dataparser."""
    def as_dict(self) -> dict:
        """Returns the dataclass as a dictionary."""
        return vars(self)

    def save_dataparser_transform(self, path: Path):
        """Save dataparser transform to json file. Some dataparsers will apply a transform to the poses,
        this method allows the transform to be saved so that it can be used in other applications.

        Args:
            path: path to save transform to
        """
        data = {
            "transform": self.dataparser_transform.tolist(),
            "scale": float(self.dataparser_scale),
        }
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with open(path, "w", encoding="UTF-8") as file:
            json.dump(data, file, indent=4)

    def transform_poses_to_original_space(
        self,
        poses: TensorType["num_poses", 3, 4],
        camera_convention: Literal["opengl", "opencv"] = "opencv",
    ) -> TensorType["num_poses", 3, 4]:
        """
        Transforms the poses in the transformed space back to the original world coordinate system.
        Args:
            poses: Poses in the transformed space
            camera_convention: Camera system convention used for the transformed poses
        Returns:
            Original poses
        """
        return transform_poses_to_original_space(
            poses,
            self.dataparser_transform,
            self.dataparser_scale,
            camera_convention=camera_convention,
        )

