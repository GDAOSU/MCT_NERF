"""
LERF configuration file.
"""

from mct.mct_dataparser import MCTDataParserConfig
from mct.mct_dataparser_sdf import MCTDataParserSDFConfig
from mct.mct_mipnerf import MCTMipNerfModel
from mct.mct_nerfacto import MCTNerfactoModelConfig
from mct.mct_neus import MCTNeuSModelConfig
from mct.mct_neusfacto import MCTNeuSFactoModelConfig
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.datamanagers.sdf_datamanager import SDFDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.neus import NeuSModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)
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
    description='Nerfacto bigwith MCT'
)


mct_method_mipnerf= MethodSpecification(
    config=TrainerConfig(
        method_name="mct_mipnerf",
        steps_per_eval_batch=5000,
        steps_per_save=5000,
        max_num_iterations=200000,
        steps_per_eval_all_images=1000000,
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
            model=VanillaModelConfig(
                _target=MCTMipNerfModel,
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
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    ),
    description='Mipnerf with MCT'
)

mct_method_neus = MethodSpecification(
    config=TrainerConfig(
        method_name="mct_neus",
        steps_per_eval_image=5000,
        steps_per_eval_batch=5000,
        steps_per_save=10000,
        steps_per_eval_all_images=1000000,  # set to a very large number so we don't eval with all images
        max_num_iterations=400000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=SDFDataManagerConfig(
                dataparser=MCTDataParserSDFConfig(),
                train_num_rays_per_batch=3000,
                eval_num_rays_per_batch=256,
                camera_optimizer=CameraOptimizerConfig(
                    mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=MCTNeuSModelConfig(
                sdf_field=SDFFieldConfig(
                    encoding_type='None', #just avoid hash encoding
                    inside_outside=False,
                ),
                background_model="none",eval_num_rays_per_chunk=1024),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
            },
            "field_background": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    ),
    description='Neus with MCT'
)

mct_method_neusfacto = MethodSpecification(
    config=TrainerConfig(
        method_name="mct_neusfacto",
        steps_per_eval_image=500,
        steps_per_eval_batch=5000,
        steps_per_save=5000,
        steps_per_eval_all_images=1000000,  # set to a very large number so we don't eval with all images
        max_num_iterations=150000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=SDFDataManagerConfig(
                dataparser=MCTDataParserSDFConfig(),
                train_num_rays_per_batch=2048,
                eval_num_rays_per_batch=1024,
                camera_optimizer=CameraOptimizerConfig(
                    mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=MCTNeuSFactoModelConfig(           
                # proposal network allows for signifanctly smaller sdf/color network
                sdf_field=SDFFieldConfig(
                    use_grid_feature=True,
                    inside_outside=False,
                    num_layers=2,
                    num_layers_color=2,
                    beta_init=0.8,
                    use_appearance_embedding=False,
                ),
                background_model="none",
                eval_num_rays_per_chunk=2048,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": MultiStepSchedulerConfig(max_steps=20001, milestones=(10000, 1500, 18000)),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20001),
            },
            "field_background": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20001),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    ),
    description='Neusacto with MCT'
)