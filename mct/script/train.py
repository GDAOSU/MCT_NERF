import glob
import os
import subprocess

import cv2


def train(id):
    cmd=["python","scripts/train.py","mct_mipnerf","--data=data/usc/area"+str(id)+"/0",
               "--output-dir=outputs/usc_area"+str(id),
               "--experiment-name=0",
               "--timestamp=0",
               "--pipeline.datamanager.dataparser.has_mask=False",
               "--steps-per-eval-image=2000000",
               "--pipeline.datamanager.train-num-images-to-sample-from=30",
               "--pipeline.datamanager.train-num-times-to-repeat-images=1000",
                "--pipeline.datamanager.eval-num-images-to-sample-from=1",
               "--pipeline.datamanager.eval-num-times-to-repeat-images=500",
               "--pipeline.datamanager.train_num_rays_per_batch=2000",
               "--max-num-iterations=150000"]
    subprocess.call(cmd)

def train_other():
    cmd=["python","scripts/train.py","mct_mipnerf","--data=data/geomvs_original/test1/0",
               "--output-dir=outputs/geomvs_original_test1",
               "--experiment-name=0",
               "--timestamp=1",
               "--pipeline.datamanager.dataparser.has_mask=False",
               "--steps-per-eval-image=200000",
               "--pipeline.datamanager.train-num-images-to-sample-from=10",
               "--pipeline.datamanager.train-num-times-to-repeat-images=1000",
                "--pipeline.datamanager.eval-num-images-to-sample-from=1",
               "--pipeline.datamanager.eval-num-times-to-repeat-images=500",
               "--pipeline.datamanager.train_num_rays_per_batch=2000",
               "--max-num-iterations=200000"]
    subprocess.call(cmd)
    
# train(2)
# train(3)
train_other()