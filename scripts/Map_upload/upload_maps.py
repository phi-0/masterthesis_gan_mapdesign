'''
This script extracts (if zipped) and uploads all map files to a Switch S3 object storage container
using the 'boto' python library see http://docs.pythonboto.org/en/latest/s3_tut.html
requires C++ 2014 build tools (visual studio or visual studio community installation)
'''

# imports
import boto3 as b3
from botocore.exceptions import ClientError
import pathlib
from config import SWITCH_S3_ACCESS_KEY, SWITCH_S3_ENDPOINT_URL, SWITCH_S3_BUCKET_NAME, SWITCH_S3_REGION, \
                   SWITCH_S3_SECRET_KEY, MAP_FOLDER_BASE_PATH, MAP_PNG_FILE_PATH

# setup connection to switch S3 storage instance
s3 = b3.resource('s3',
                 region_name=SWITCH_S3_REGION,
                 endpoint_url=SWITCH_S3_ENDPOINT_URL,
                 aws_access_key_id=SWITCH_S3_ACCESS_KEY,
                 aws_secret_access_key=SWITCH_S3_SECRET_KEY
                 )

# check connection
for bucket in s3.buckets.all():
    print(bucket.name)


# get local images to upload
fpath = r'{}'.format(MAP_FOLDER_BASE_PATH)
data_dir = pathlib.Path(fpath + MAP_PNG_FILE_PATH)
imgs = list(data_dir.glob('*.png'))

print(len(imgs))

# upload images
s3 = s3.Bucket(SWITCH_S3_BUCKET_NAME)
for i in range(len(imgs)):
    try:
        message = s3.upload_file(str(imgs[i]), f'maps/map_{str(i+1)}.png')
        print(f'[Image {i}] Successfully uploaded {imgs[i]}')
    except ClientError as e:
        print(f'Error while uploading image {imgs[i]} \n\n The following error was thrown: {e}')

