#! /bin/bash
# uploads a selected number of images while choosing a random number between 1-10 to select a single image per group

for imagenum in $@
do
    rnd=$((1 + ${RANDOM} % 10))
    echo "uploading sample_image_epoch$imagenum-$rnd.png --> s3://storage_hil/output/sample$imagenum.png"
    sudo s3cmd put /data/output/images/uploading sample_image_epoch$imagenum-$rnd.png.png s3://storage_hil/output/sample$imagenum.png
done