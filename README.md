# GAN-based Map Design - *DwarfGAN*


## Background
This repo contains the scripts, models and further required setup documents created as part of the master's thesis at the HSLU for MSc in Applied Information and Data Science.


## Data
Data used for this master's thesis is based on the public, player maintained Dwarf Fortress Map Archive ([DFMA](https://mkv25.net/dfma/)). In order to help share and preserve player generated maps a corresponding GitHub archive was established ([DFMA Map Archive on GitHub](https://github.com/df-map-archive/dfma-map-file-archive)). This GitHub archive served as the basis for the use case implementation of this thesis.

The extracted sample map images (see section **Pre-Processing**) sourced from the DFMA GitHub archive were uploaded to a public S3 bucket accessible [HERE](https://os.zhdk.cloud.switch.ch/swift/v1/storage_hil/). To download a single map image simple append the chosen map ID (valid IDs: **[1,119563]**) to the URL of the S3 bucket in the following format:

>    os.zhdk.cloud.switch.ch/swift/v1/storage_hil/**maps/map_[ID].png**

Three example input map images are shown below (varying input dimensions upto 6000 x 6000 pixels):

![Example Input Map 1](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/data/input_samples/map_5.png)
![Example Input Map 2](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/data/input_samples/map_358.png)
![Example Input Map 3](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/data/input_samples/map_78.png)


## Pre-Processing (model external)
1. Source ([DFMA Map Archive on GitHub](https://github.com/df-map-archive/dfma-map-file-archive) with [GitHub LFS](https://git-lfs.github.com/))
1. Extract Map Archive *.FDF-MAP* files ([GUI-Automation script](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/scripts/DF_map_conversion/auto_conversion.py))
1. Upload of extracted source maps to S3 bucket ([Python script with *boto3*](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/scripts/Map_upload/upload_maps.py))
1. Random Crop and Filter ([Python pre-processing scripts](https://github.com/phi-0/masterthesis_gan_mapdesign/tree/master/scripts/image_preprocessing)). The different models require differently sized input crops. Generally the input and output dimensions of the model trained as part of this thesis are included in the model name (e.g. WGAN-GP**256**)
1. Later Models (WGAN-GP RUN06 onwards): 
    1. semi-automated selection of ASCII-based maps ([Jupyter notebook](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/scripts/image_preprocessing/ASCII_subset.ipynb)) 
    1. structured cropping to extract tiles from tilesets ([Tileset crop script](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/scripts/image_preprocessing/tileset_crop.py) used on the original *DwarfFortress* [ASCII tilesets](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/data/tiles/standard_tileset.png) to extract [single tiles](https://github.com/phi-0/masterthesis_gan_mapdesign/tree/master/data/tiles/800x600)) and to crop input map images to 12x12 dimensional single tiles.

Further pre-processing was implemented as part of the models using keras pre-processing layers (**rescaling** and - if required - **resizing**).

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



## Output

Trained models are saved as *.h5* files and their last training state is shared via GitHub ([models](https://github.com/phi-0/masterthesis_gan_mapdesign/tree/master/models)). Model saves of later, bigger models are massive (upto 400MB) and cannot be shared via GitHub (120MB single file limit). All Models have been shared via S3 bucket ([S3 bucket - models](https://os.zhdk.cloud.switch.ch/swift/v1/storage_hil/)). In order to download a single model save file add the folder name */models*, the model folder (e.g. *DCGAN256*) and the model save file name (see list below)

>  https://os.zhdk.cloud.switch.ch/swift/v1/storage_hil/**models/DCGAN256/generator-2021-03-24_174741.h5**

### **Available Model Saves**

- [models/DCGAN1024/](https://os.zhdk.cloud.switch.ch/swift/v1/storage_hil/models/DCGAN1024/generator-2021-03-17_104329.h5)
- [models/DCGAN256/](https://os.zhdk.cloud.switch.ch/swift/v1/storage_hil/models/DCGAN256/generator-2021-03-24_174741.h5)
- [models/WGANGP-RUN01/](https://os.zhdk.cloud.switch.ch/swift/v1/storage_hil/models/WGANGP-RUN01/generator-2021-03-31_180243.h5)
- [models/WGANGP-RUN02/](https://os.zhdk.cloud.switch.ch/swift/v1/storage_hil/models/WGANGP-RUN02/generator-2021-04-04_025322.h5)
- [models/WGANGP-RUN03/](https://os.zhdk.cloud.switch.ch/swift/v1/storage_hil/models/WGANGP-RUN03/generator-2021-04-09_011627.h5)
- [models/WGANGP-RUN05/](https://os.zhdk.cloud.switch.ch/swift/v1/storage_hil/models/WGANGP-RUN05/generator-2021-04-17_090503.h5)
- [models/WGANGP-RUN06/](https://os.zhdk.cloud.switch.ch/swift/v1/storage_hil/models/WGANGP-RUN06/generator-2021-04-20_050013.h5)
- [models/WGANGP-RUN07/](https://os.zhdk.cloud.switch.ch/swift/v1/storage_hil/models/WGANGP-RUN07/generator-2021-04-22_230202.h5)
- [models/WGANGP-RUN08/](https://os.zhdk.cloud.switch.ch/swift/v1/storage_hil/models/WGANGP-RUN08/generator-2021-05-01_091206.h5)

### **Generating New Maps**

To generate new map images a separate jupyter notebook is included [HERE](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/notebooks/Generate_Map.ipynb). Once the model saves above have been downloaded and are accessible to the docker container running the jupyter lab server, the paths to the model save folders need to be adjusted at the top of the first cell (*models* dictionary). After that the script cells can be executed and will ask for a model name which will then generate a single output map image based on that learned model. The GIF below shows the mechanism in action:

![](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/output/images/generate_map_animated.gif)