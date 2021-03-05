#! /bin/bash
# uploads a selected number of images while choosing a random number between 1-10 to select a single image per group

for imagenum in $@
do
    echo "uploading sample_image_epoch$imagenum-$((1 + $RANDOM % 10)).png --> s3://storage_hil/output/sample$imagenum.png"
    sudo s3cmd put /data/output/images/sample_image_epoch$imagenum-$((1 + $RANDOM % 10)).png s3://storage_hil/output/sample$imagenum.png
done