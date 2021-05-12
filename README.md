# GAN-based Map Design - *DwarfGAN*


## Background
This repo contains the scripts, models and further required setup documents created as part of the master's thesis at the HSLU for MSc in Applied Information and Data Science.


## Data
Data used for this master's thesis is based on the public, player maintained Dwarf Fortress Map Archive ([DFMA](https://mkv25.net/dfma/)). In order to help share and preserve player generated maps a corresponding GitHub archive was established ([DFMA Map Archive on GitHub](https://github.com/df-map-archive/dfma-map-file-archive)). This GitHub archive served as the basis for the use case implementation of this thesis.

The extracted sample map images (see section **Pre-Processing**) sourced from the DFMA GitHub archive were uploaded to a public S3 bucket accessible [HERE](https://os.zhdk.cloud.switch.ch/swift/v1/storage_hil/). To download a single map image simple append the chosen map ID (valid IDs: **[1,119563]**) to the URL of the S3 bucket in the following format:

>    os.zhdk.cloud.switch.ch/swift/v1/storage_hil/**maps/map_[ID].png**

Three example input map images are shown below:

![Example Input Map 1](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/data/input_samples/map_5.png)
![Example Input Map 2](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/data/input_samples/map_358.png)
![Example Input Map 3](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/data/input_samples/map_78.png)


## Pre-Processing (model external)
1. Source ([DFMA Map Archive on GitHub](https://github.com/df-map-archive/dfma-map-file-archive) with [GitHub LFS](https://git-lfs.github.com/))
1. Extract Map Archive *.FDF-MAP* files ([GUI-Automation script](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/scripts/DF_map_conversion/auto_conversion.py))
1. Upload of extracted source maps to S3 bucket ([Python script with *boto3*](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/scripts/Map_upload/upload_maps.py))
1. Random Crop and Filter ([Python pre-processing scripts](https://github.com/phi-0/masterthesis_gan_mapdesign/tree/master/scripts/image_preprocessing))
1. Later Models (WGAN-GP RUN06 onwards): 
    1. semi-automated selection of ASCII-based maps ([Jupyter notebook](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/scripts/image_preprocessing/ASCII_subset.ipynb)) 
    1. structured cropping to extract tiles from tilesets ([Tileset crop script](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/scripts/image_preprocessing/tileset_crop.py) used on the original *DwarfFortress* [ASCII tilesets](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/data/tiles/standard_tileset.png) to extract [single tiles](https://github.com/phi-0/masterthesis_gan_mapdesign/tree/master/data/tiles/800x600))



## Setup

This repo contains the [Dockerfile](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/Dockerfile) required to create a docker container able to run all jupyter notebooks in [./notebooks](https://github.com/phi-0/masterthesis_gan_mapdesign/tree/master/notebooks). The docker file is based on a tensorflow image including GPU support, provided the necessary drivers and tools ([*Docker*](https://docs.docker.com/engine/install/ubuntu/), [*CUDA drivers*](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions), [*NVIDIA Container Toolkit*](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)) are available on the host machine. Building the docker container with the above Dockerfile will further clone this repo and install python dependencies (jupyterlab, keras, boto3).

Building the container (in a folder with only the Dockerfile):

> *docker build . --no-cache -t 'tf-gpu'*

Run the container with GPU support, port-forwarding for JupyterLab and Tensorboard ports, mapping the partitions containing input and output data (*data* and *data2*) and attach to running container with *bash*:
> *sudo docker run --rm --gpus all -v /data:/data /data2:/data2 -it -p 8888:8888 -p 6006:6006 -p 8061:8061 --name dwarfgan tf-gpu bash*

Once inside the docker container initialize the Jupyter Lab server based on root directory of container:
> *jupyter lab --ip 0.0.0.0 --allow-root --notebook-dir=/*

To start the TensorBoard demon inside jupyter lab, open a new console and run the following with *--logdir=* set to the drive and folder supposed to containe the tensorboard checkpoints:
> *tensorboard --port=8061 --logdir=/data/output/tensorboard --bind_all*
