import pathlib
import urllib.request
import os

'''
*****************************
Configuration
'''

# SPECIFY THE OUTPUT PATH OR ACCEPT DEFAULT = [WORKING_DIRECTORY]/data/
output_path = None

# SPECIFY THE NUMBER OF SAMPLES TO BE DOWNLOADED (MAX = 119'563) OR ACCEPT DEFAULT = 1000
n_samples = None

'''
*****************************
'''

# Get current working directory
wd = os.getcwd()

# Define output path and specify subfolder '/maps' (required for keras "load images from directory" layer)
if output_path is None:
    output_path = pathlib.Path(f'{wd}/data/maps')
else:
    output_path = pathlib.Path(f'{output_path}/maps')


def download_maps(out=output_path, n=1000):
    '''
    
    Downloads the first n maps from the S3 bucket 
    https://os.zhdk.cloud.switch.ch/swift/v1/storage_hil/
    to the specified output_path

    '''
    #check if output folders exists, else create
    if not out.exists():
        if not out.parents[0].exists(): # if output folder is manually defined, check if the specified folder exists
            os.mkdir(out.parents[0])    # create 'data' directory if no directory is manually specified
        os.mkdir(out)                   # create 'data/maps' directory
    else:
        pass

    for map in range(1,n+1):
        url = f'https://os.zhdk.cloud.switch.ch/swift/v1/storage_hil/maps/map_{map}.png'
        urllib.request.urlretrieve(url, f'{out}/map_{map}.png')
        print(f'Finished downloading sample map: {url}')


# Execute function
if n_samples is None or n_samples > 119563:
    download_maps(output_path)
else:
    download_maps(output_path, n_samples)