from __future__ import annotations

from pathlib import Path

import torch

from nerfstudio.configs.base_config import Config
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.nerfacto_uniformsample import NerfactoUniformModelConfig
from nerfstudio.models.semantic_nerfw import SemanticNerfWModelConfig
from nerfstudio.models.tensorf import TensoRFModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    # for buffer in model.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb
    # print('model size: {:.3f}MB'.format(size_all_mb))


def get_params_size(params):
    param_size = 0
    for key in params:
        param_list = params[key]
        for param in param_list:
            param_size += param.nelement() * param.element_size()

    size_all_mb = (param_size) / 1024**2
    return size_all_mb


def print_model_size():
    aabb = SceneBox()
    aabb.aabb = torch.tensor([[-1, -1, -1], [1, 1, 1]])

    model = method_configs["mipnerf"].pipeline.model.setup(scene_box=aabb, sampling_box=aabb, num_train_data=1)
    model_size = get_params_size(model.get_param_groups())
    print("{} size: {:.3f}MB".format("mipnerf", model_size))

    model = method_configs["vanilla-nerf"].pipeline.model.setup(scene_box=aabb, sampling_box=aabb, num_train_data=1)
    model_size = get_params_size(model.get_param_groups())
    print("{} size: {:.3f}MB".format("vanilla-nerf", model_size))

    model = method_configs["tensorf"].pipeline.model.setup(scene_box=aabb, sampling_box=aabb, num_train_data=1)
    model_size = get_params_size(model.get_param_groups())
    print("{} size: {:.3f}MB".format("tensorf", model_size))

    model = method_configs["dortmund"].pipeline.model.setup(scene_box=aabb, sampling_box=aabb, num_train_data=1)
    model_size = get_params_size(model.get_param_groups())
    print("{} size: {:.3f}MB".format("dortmund", model_size))

    model = method_configs["dortmund-whole"].pipeline.model.setup(scene_box=aabb, sampling_box=aabb, num_train_data=1)
    model_size = get_params_size(model.get_param_groups())
    print("{} size: {:.3f}MB".format("dortmund-whole", model_size))

    model = method_configs["nerfacto"].pipeline.model.setup(scene_box=aabb, sampling_box=aabb, num_train_data=1)
    model_size = get_params_size(model.get_param_groups())
    print("{} size: {:.3f}MB".format("nerfacto", model_size))

    model = method_configs["instant-ngp"].pipeline.model.setup(scene_box=aabb, sampling_box=aabb, num_train_data=1)
    model_size = get_params_size(model.get_param_groups())
    print("{} size: {:.3f}MB".format("instant-ngp", model_size))


print_model_size()
