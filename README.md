# GAN-based Map Design - DwarfGAN


## Background
This repo contains the scripts, models and further required setup documents created as part of the master's thesis at the HSLU for MSc in Applied Information and Data Science.


## Data
Data used for this master's thesis is based on the public, player maintained Dwarf Fortress Map Archive ([DFMA](https://mkv25.net/dfma/)). In order to help share and preserve player generated maps a corresponding GitHub archive was established ([DFMA Map Archive on GitHub](https://github.com/df-map-archive/dfma-map-file-archive)). This GitHub archive served as the basis for the use case implementation of this thesis.

The extracted sample map images (see section **Pre-Processing**) sourced from the DFMA GitHub archive were uploaded to a public S3 bucket accessible [HERE](https://os.zhdk.cloud.switch.ch/swift/v1/storage_hil/). To download a single map image simple append the chosen map ID (valid options: **[1,119563]**) to the URL of the S3 bucket in the following format:

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
1. Later Models: 
    1. semi-automated selection of ASCII-based maps ([Jupyter notebook](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/scripts/image_preprocessing/ASCII_subset.ipynb)) 
    1. structured cropping to extract tiles from tilesets ([Tileset crop script](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/scripts/image_preprocessing/tileset_crop.py) used on the original *DwarfFortress* [ASCII tilesets](https://github.com/phi-0/masterthesis_gan_mapdesign/blob/master/data/tiles/standard_tileset.png) to extract [single tiles](https://github.com/phi-0/masterthesis_gan_mapdesign/tree/master/data/tiles/800x600))