import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.exporter.exporter_utils import generate_point_cloud_all_mct
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup


def generate_pcd(trained_model_dir,num_pts,out_dir):
    trained_blocks=glob.glob(os.path.join(trained_model_dir,"*"))
    num_blks=len(trained_blocks)
    num_pts_per_blk=int(num_pts/num_blks)
    for id,block in enumerate(trained_blocks):
        _, pipeline, _, _ = eval_setup(Path(os.path.join(block,"mct_mipnerf/0/config.yml")),1024,test_mode='all')

        # Increase the batchsize to speed up the evaluation.
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = 1024

        scale_input2nerf=pipeline.datamanager.train_dataparser_outputs.dataparser_scale
        shift_input2nerf=pipeline.datamanager.train_dataparser_outputs.dataparser_shift
        shift=-shift_input2nerf.numpy()

        scale=(1/scale_input2nerf).numpy()
        #offset=np.array([-self.offx,-self.offy,-self.offz])
        #shift+=offset
        pcd = generate_point_cloud_all_mct(
            pipeline=pipeline,
            rgb_output_name="rgb_fine",
            depth_output_name="depth_fine",
            output_dir=Path(out_dir),
            num_pts=num_pts_per_blk,
            shiftx=shift[0],
            shifty=shift[1],
            shiftz=shift[2],
            scale=scale
        )
        o3d.t.io.write_point_cloud(os.path.join(out_dir, str(id)+".ply"), pcd)
        torch.cuda.empty_cache()
                    


generate_pcd(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\dortmund_metashape_blocks64',10000000,
             r'J:\xuningli\cross-view\ns\nerfstudio\pcd\dortmund_blocks16')


