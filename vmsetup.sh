#! /bin/bash

echo "Setting up data mount..."
echo "--------------------------------------"
mkdir /data
mount -t ext3 /dev/vdb1 /data

echo "create startup script to remap after reboot"
sudo bash -c 'echo "#! /bin/bash" > /etc/init.d/startup_mount.sh'
sudo bash -c 'echo "sudo mount -t ext3 /dev/vdb1 /data" >> /etc/init.d/startup_mount.sh'

echo "change permission of startup script"
sudo chmod +x /etc/init.d/startup_mount.sh

echo "Installing Docker..."
echo "--------------------------------------"
apt-get update
apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

echo "Check fingerprint: $(apt-key fingerprint 0EBFCD88)"

add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io

echo "Installing NVIDIA container toolkit..."
echo "--------------------------------------"

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \ 
&& curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
&& curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update
apt-get install -y nvidia-docker2

echo "Restarting docker runtime"
systemctl restart docker

echo "Checking NVIDIA docker installation"
docker run --rm --gpus all nvidia/cuda:10.0-base nvidia-smi