# About
This method is based on nerfstudio repo.

# Quickstart

## 1. Installation: Setup the environment
### Prerequisites
You must have an NVIDIA video card with CUDA installed on the system. This library has been tested with version 11.3 of CUDA. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)
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

