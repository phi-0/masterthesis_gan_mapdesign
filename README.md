# GAN based Map Design - DwarfGAN


## Background
This repo contains the scripts, models and further required setup documents created as part of the master's thesis at the HSLU for MSc in Applied Information and Data Science.


## Data
Data used for this master's thesis is based on the public, player maintained Dwarf Fortress Map Archive ([DFMA](https://mkv25.net/dfma/)). In order to help share and preserve player generated maps a corresponding GitHub archive was established ([DFMA Map Archive on GitHub](https://github.com/df-map-archive/dfma-map-file-archive)). This GitHub archive served as the basis for the use case implementation of this thesis.

The extracted sample map images (see section **Pre-Processing**) sourced from the DFMA GitHub archive were uploaded to a public S3 bucket accessible [HERE](https://os.zhdk.cloud.switch.ch/swift/v1/storage_hil/). To download a single map image simple append the chosen map ID (valid options: **[1,119563]**) to the URL of the S3 bucket in the following format:

>    os.zhdk.cloud.switch.ch/swift/v1/storage_hil/**maps/map_[ID].png**

Three example input map images are shown below:

![Example Input Map 1](https://os.zhdk.cloud.switch.ch/swift/v1/storage_hil/maps/map_5.png)
![Example Input Map 2](https://os.zhdk.cloud.switch.ch/swift/v1/storage_hil/maps/map_358.png)
![Example Input Map 3](https://os.zhdk.cloud.switch.ch/swift/v1/storage_hil/maps/map_78.png)
