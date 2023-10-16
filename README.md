# Enabling Neural Radiance Fields for Large-scale Aerial Images – from the 3D Geometric Reconstruction Perspective  
## Ningli Xu, Rongjun Qin, Debao Huang, Fabio Remondino  
## Abstract  
Neural Radiance Fields (NeRF) have recently emerged as a novel and effective 3D scene representation model, which has the potential to benefit typical 3D reconstruction tasks such as aerial photogrammetry. However, as a learning-based approach, its scalability and the accuracy of the inferred 3D geometry are not well examined when applied to large-scale aerial assets. This study aims to provide a thorough assessment of NeRF in 3D reconstruction from aerial images and compare the results with results generated from traditional multi-view stereo (MVS) pipelines. Typically, NeRF is computation-heavy and requires a specific sampling strategy when scaling to large datasets. We observe that the conventional random sampling strategy results in slower convergence rates as the dataset size increases. In response, we present an alternative approach by introducing a location-specific sampling technique. This novel method involves the dynamic allocation of training resources based on spatial positions, leading to accelerated convergence rates and lessened memory consumption. To facilitate implementing this technique into existing NeRF frameworks, we introduce a multi-camera tiling (MCT) method. The idea is to decompose a large-frame image modeled by a single pinhole model, into multiple tiled images with different pinhole models with varying intrinsic parameters, in that these small-frame images can be fed into a NeRF training process on the demand of specifically sampled locations without the loss of resolution. We compare this variant with three traditional MVS pipelines on typical photogrammetric aerial block datasets against ground truth reference data. Both qualitative and quantitative results suggest that the evaluated NeRF approach produces better completes and object details than traditional approaches, while as of now, still fall short in accuracy.   
![image](https://github.com/GDAOSU/MCT_NERF/assets/32317924/e306b365-4083-4905-a2b8-ec060363ac9b)
![image](https://github.com/GDAOSU/MCT_NERF/assets/32317924/a217ccb2-bac7-467a-a3e6-cf0cb016b49e)
![image](https://github.com/GDAOSU/MCT_NERF/assets/32317924/b7b54d3b-4732-4a2e-ba2a-cf1953945a96)

# Updates and To Do 
- [ ] Release the sample datasets, pretrained models   
- [ ] Speed up the rendering RGB,Depth,point cloud, mesh  
- [x] [08/30/2023] Release the code.  


# Quickstart

## 1. Installation: Setup the environment
### Prerequisites
You must have an NVIDIA video card with CUDA installed on the system. This library has been tested with version 11.3 of CUDA. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).
If you try to install pytorch with CUDA 11.7, it is not necessary to install CUDA individually.
### Create environment
Nerfstudio requires `python >= 3.7`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/en/latest/miniconda.html) before proceeding.
```bash
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip
```
### Dependencies
Install pytorch with CUDA (this repo has been tested with CUDA 11.3 and CUDA 11.7) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
For CUDA 11.3:
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
For CUDA 11.7:
```bash
pip install torch==1.13.1 torchvision functorch --extra-index-url https://download.pytorch.org/whl/cu117
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
See [Dependencies](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md#dependencies)
in the Installation documentation for more.
### Installing nerfstudio
```bash
git clone [https://github.com/GDAOSU/MCT_NERF/edit/mct/README.md](https://github.com/GDAOSU/MCT_NERF)
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
```
### Installing mct module
```bash
cd mct
pip install -e .
```
# Datasets
Currently, we only support dataset in colmap format  
# Training & Rendering
## Step1: preprocess   
use multi-camera tiling to crop large high-res images into smaller images by spliting whole scene into many blocks  
```bash
python mct/script/step1_preprocess.py
```

## Step2: block training  
training each blocks  
```bash
python mct/script/step2_batch_train.py
```

## Step3: Rendering images or point cloud  
rendering the novel view or dense point cloud   
```bash
python mct/script/step3_generate_pcd.py  
```
```bash
python mct/script/step3_render_image.py  
```

