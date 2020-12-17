FROM tensorflow/tensorflow:2.3.1-gpu-jupyter

RUN apt-get update

RUN apt-get -y install git

RUN cd ~/

RUN git clone https://github.com/phi-0/masterthesis_gan_mapdesign.git

RUN apt-get install -y python3.8 python3-pip

#RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

#RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64
#RUN pip3 install -r ./masterthesis_gan_mapdesign/requirements.txt