"""
LERF configuration file.
"""

from mct.mct_dataparser import MCTDataParserConfig
from mct.mct_nerfacto import MCTNerfactoModelConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

mct_method_nerfacto= MethodSpecification(
    config=TrainerConfig(
        method_name="mct_nerfacto",
        steps_per_eval_batch=5000,
        steps_per_save=5000,
        max_num_iterations=30000,
        steps_per_eval_all_images=10000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=MCTDataParserConfig(),
                train_num_images_to_sample_from=240,
                train_num_times_to_repeat_images=1000,
                eval_num_images_to_sample_from=240,
                eval_num_times_to_repeat_images=1000,
                train_num_rays_per_batch=2048,
                eval_num_rays_per_batch=2048,
                camera_optimizer=CameraOptimizerConfig(
                    mode="off",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                    scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
                ),
            ),
            model=MCTNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    ),
    description='Nerfacto with MCT'
)

mct_method_nerfacto_big= MethodSpecification(
    config=TrainerConfig(
        method_name="mct_nerfacto_big",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="off",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                    scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
                ),
            ),
            model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    ),
    description='Nerfacto large model with MCT'
)

mct_method_mipnerf= MethodSpecification(
    config=TrainerConfig(
        method_name="mct_mipnerf",
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(dataparser=NerfstudioDataParserConfig(), train_num_rays_per_batch=1024),
            model=VanillaModelConfig(
                _target=MipNerfModel,
                loss_coefficients={"rgb_loss_coarse": 0.1, "rgb_loss_fine": 1.0},
                num_coarse_samples=128,
                num_importance_samples=128,
                eval_num_rays_per_chunk=1024,
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            }
        },
    ),
    description='Mipnerf with MCT'
)