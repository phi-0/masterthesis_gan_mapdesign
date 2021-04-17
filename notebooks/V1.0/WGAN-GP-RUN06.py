#!/usr/bin/env python
# coding: utf-8

# # DwarfGAN - Deep Learning based Map Design for Dwarf Fortress

# ### Setup and Imports

# In[1]:


""" 

loosely based on example coded provided in Foster, 2019 see:

basic GAN
"G:\Dev\DataScience\GDL_code\models\GAN.py"

Wasserstein GAN
"G:\Dev\DataScience\GDL_code\models\WGAN.py"

Wasserstein GAN with Gradient Penatly
"G:\Dev\DataScience\GDL_code\models\WGANGP.py"

"""
# imports
from keras.layers import Add, Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D, LayerNormalization
from keras.layers.experimental import preprocessing
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.initializers import RandomNormal
from keras.losses import binary_crossentropy, Loss
from keras import metrics
from functools import partial
from tensorflow import keras

import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import PIL
import PIL.Image
import pathlib
import tensorflow as tf
import io
import random

#!pip install boto3
import boto3 as b3 


print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

############### CONFIG ###################

# model name
model_name = 'dwarfganWGANGPR06'
# folder path to input files (map images)
fpath = r'/data2/input'
# folder path to tensorboard output
tboard_dir = '/data2/output/tensorboard'
# folder path for saved model output
out_model_dir = '/data2/output/models'
# folder for images to be saved during training
out_img_dir = '/data2/output/images'
# frequency of checkpoint saves (images, model weights) in epochs
CHECKPOINT = 50
# use skip connections (additive)?
SKIP = False
LATENT_DIM = 128
EPOCHS = 1000 
#BATCH_PER_EPOCH = 20
# pre-processed (cropped) images are 1024x1024. We will later resize the images to 256x256 due to memory restrictions.
IMAGE_SIZE = (128,128)
BATCH_SIZE = 128
CRITIC_FACTOR = 5 # number of times the critic is trained more often than the generator. Recommended = 5
GRADIENT_PENALTY_WEIGHT = 10
RELU_SLOPE_C = 0.2
RELU_SLOPE_G = 0.2
DROPOUT_C = 0.3
MOMENTUM_G = 0.9
CRIT_LR = 0.0003 # Adjusted learning rates according to two time-scale update rule (TTUR), see Heusel et al., 2017
GEN_LR = 0.0001

# NOTE: all extracted map PNGs have been saved on a separate virtual disk mapped to '/data' of the virtual machine in use
data_dir = pathlib.Path(fpath + '/maps')
imgs = list(data_dir.glob('*.png'))

##########################################


# ### Data Load and Train / Validation Split

# The map images sourced from the DFMA come in a variety of dimensions. In order to create sample images with constant dimensions, as required by tensors, the 100k input samples were run through a python script to randomly crop 10 1024 x 1024 areas per picture. Of those cropped (sub-)images, only the ones which contain structures were retained. This was achieved by filtering out image crops which only contained two or less different colors. With that, the logic mainly filterd out crops which only contained black. This process resulted in 700'000+ (sub-)image samples.

# In[2]:


# use pre-processed (cropped) 128 x 128 images
data_dir = pathlib.Path(fpath + '/ascii_crops_128/maps')
imgs = list(data_dir.glob('*.png'))
print(f'There are {str(len(imgs))} cropped image samples available')


# In[3]:


# show example sample image (cropped to 128x128)
print('A random sample input image:')
PIL.Image.open(imgs[random.randint(0,len(imgs))])


# In[3]:




# creating keras datasets for training and validation - refer to https://www.tensorflow.org/tutorials/images/classification
dataset_train = tf.keras.preprocessing.image_dataset_from_directory(  fpath+'/ascii_crops_128',
                                                                      image_size=IMAGE_SIZE, 
                                                                      batch_size=BATCH_SIZE, 
                                                                      labels=[1.] * len(imgs), # setting all labels to 1.0 (for 'real') as float32
                                                                      #label_mode=None, # yields float32 type labels
                                                                      seed=543 #,
                                                                      #validation_split=0.8, # due to the large number of images, we split out more into the validation set (which is not being used)
                                                                      #subset='training'
                                                                    )

#dataset_val = tf.keras.preprocessing.image_dataset_from_directory(  fpath,
#                                                                    image_size=IMAGE_SIZE, 
#                                                                    batch_size=BATCH_SIZE, 
#                                                                    labels=[1.] * len(imgs), # setting all labels to 1.0 (for 'real') as float32
#                                                                    #label_mode=None, # yields float32 type labels
#                                                                    seed=42,
#                                                                    validation_split=0.5,
#                                                                    subset='validation'
#                                                                    )


# refer to https://www.tensorflow.org/tutorials/images/classification
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = dataset_train.cache().prefetch(buffer_size=BATCH_SIZE)
#val_ds = dataset_val.cache().prefetch(buffer_size=BATCH_SIZE)
#val_ds = None


# In[4]:


N_TRAINING = 23227 # = 20% of total samples


# ### Check Random Sample from Dataset

# In[29]:

'''
# Check random images from prepared batches
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1): # take one batch. Here batch_size = 128 examples per batch
    for i in range(9): # show first 9 images of batch
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(f'Image Check {i+1}, label {labels[i]}')
        plt.axis('off')
'''


# # UPDATED Model Definition
# 
# Generally the following changes have been implemented to the architectures:
# 
# ### RUN04
# - increased number of filters
# - instead of stagnating number of filters, the number of filters now steadily decreases (generator) / increases (discriminator). The last CONV layer of the discriminator would have 1024 filters however that results in a large number of nodes on the last DENSE layer and therefore an increase of trainable parameters by around 7 milion. Therefore only 512 filters are configured for the the last discriminator CONV layer.
# - changed dimension of first dense layer in generator after reshape to (8,8,1024) to provide more features to be learned (see Radford et al., 2015)
# - deactivated biases in first DENSE layer and CONV2DTranspose layers of generator 
# - implemented dropout for all discriminator layers expect first and last
# - added batchnorm(0.9) for all generator layers (default is 0.99)
# 
# The following changes have been implemented to the architecture for RUN04:
# 
# - changed input image crops to 128 x 128
# - changed critic learning rate to 0.0003 (from 0.0002) so now larger than generator learning rate 0.0002)
# - increase latent dimension to 128 (from 100)
# 
# ### RUN05
# 
# general:
# - filter size: reviewing available tilesets, it became clear that most glyphs in the available tilesets are bigger than the current filter (5x5). So filter size from earlier filters (ie closer to the input/output image) were increased to [16x16], 12x12, 8x8 filters for the first/last 3 conv layers
# 
# training:
# 
# - based on Brock et al. (BigGAN) the training batch size was increased to 128
# - so far, different kinds of tilesets were mixed in with the training set. Now only the original ASCII-based tileset sample were retained (manual selection process)
# - tensorboard callback implemented
# 
# generator:
# - increase initial layer dimensions to (4,4,2048)
# - added additional convolutional upscaling layer between first (4,4,2048) layer to (8,8,1024)
# - reactivated 64 filter layer between 64x64x128 and final 128x128x3 layer with kernel size 1
# - previously, the first convolutional layer in the generator model had a different kernel size (3) VS the other conv layers (5). Now all layers have the same kernel size = 5
# - momentum decrease to 0.9 (from 0.99)
# - batchnorm inserted after tanh() activation of last layer (before rescaling)
# 
# critic:
# - added a dropout layer between flatten and output layer (last Dense(1) layer)
# 
# 
# ### RUN06
# 
# critic:
# - first kernel adjusted to size (10x12), the approximate size of default ASCII tokens
# - increased number of filters for first conv2d layer to 32-->64
# 
# generator:
# - adjusted filter size of last generator Conv2DTranspose layer to output 10x12 tokens (kernel_size=(10,12)) with the stride adjusted so there is no overlap (stride=(12,10)).
# - added skip connection logic (additive) to generator. For RUN06 this was deactivated (skip = False)
# - try layer norm along axis -1 (pixelwise normalization see ProGAN paper) for last generator normalization layer
# 
# 
# ### Discriminator

# In[5]:




def discriminator_model():

    # DISCRIMINATOR
    # set input variables to variable width + height. Will be cropped in preprocessing [CURRENTLY FIXED TO 256x256]
    input_dim = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

    # Input
    d_input = Input(shape=input_dim, name='Discriminator_Input')

    # ---- REMOVED FOR 256x256 NETWORK ----------
    # Keras-based preprocessing. Alternative: RandomCrop()
    # use smart_resizing?
    #x = tf.keras.preprocessing.image.smart_resize(d_input, (1024, 1024))
    #x = preprocessing.Resizing(width=512, 
    #                           height=512, 
    #                           name='Preprocessing_Resize'
    #                          )(d_input) # Resize to 512 x 512 images

    #x = preprocessing.RandomCrop(height=512, 
    #                            width=512, 
    #                            name = 'Preprocessing_RandomCrop'
    #                           )(d_input)

    x = preprocessing.Rescaling(scale=1./127.5, 
                                offset=-1,
                                name='Preprocessing_Rescale'
                               )(d_input) # Rescale values from [0,255] to [-1,1] see https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Rescaling


    # Conv2D Layer 0
    x = Conv2D(
            filters = 64,
            kernel_size = (12,10), 
            strides = 1,
            padding = 'same',
            kernel_initializer = RandomNormal(mean=0., stddev=0.02),
            name = 'Discriminator_Conv2D_Layer_0'
    )(x)
    
    # Activation 0 - Leaky ReLU
    x = LeakyReLU(alpha = RELU_SLOPE_C, name='Activation_0')(x)
    
    
    # Conv2D Layer 1
    x = Conv2D(
            filters = 64,
            kernel_size = 5,
            strides = 2,
            padding = 'same',
            kernel_initializer = RandomNormal(mean=0., stddev=0.02),
            name = 'Discriminator_Conv2D_Layer_1'
    )(x)

    # BatchNorm Layer 1
    #x = BatchNormalization()(x)

    # Activation 1 - Leaky ReLU
    x = LeakyReLU(alpha = RELU_SLOPE_C, name='Activation_1')(x)

    # Dropout 1
    x = Dropout(rate = DROPOUT_C)(x)


    # Conv2D Layer 2
    x = Conv2D(
            filters = 128,
            kernel_size = 5,
            strides = 2,
            padding = 'same',
            name = 'Discriminator_Conv2D_Layer_2',
            kernel_initializer = RandomNormal(mean=0., stddev=0.02)
    )(x)

    # BatchNorm Layer 2
    #x = BatchNormalization()(x)

    # Activation 2 - Leaky ReLU
    x = LeakyReLU(alpha = RELU_SLOPE_C, name='Activation_2')(x)

    # Dropout 2
    x = Dropout(rate = DROPOUT_C)(x)


    # Conv2D Layer 3
    x = Conv2D(
            filters = 256,
            kernel_size = 5,
            strides = 2,
            padding = 'same',
            name = 'Discriminator_Conv2D_Layer_3',
            kernel_initializer = RandomNormal(mean=0., stddev=0.02)
    )(x)

    # BatchNorm Layer 3
    #x = BatchNormalization()(x)

    # Activation 3 - Leaky ReLU
    x = LeakyReLU(alpha = RELU_SLOPE_C, name='Activation_3')(x)

    # Dropout 3
    x = Dropout(rate = DROPOUT_C)(x)



    # Conv2D Layer 4
    x = Conv2D(filters = 512,
               kernel_size = 5,
               strides = 2,
               padding = 'same',
               name = 'Discriminator_Conv2D_Layer_4',
               kernel_initializer = RandomNormal(mean=0., stddev=0.02)
    )(x)

    # BatchNorm Layer 4
    #x = BatchNormalization()(x)

    # Activation 4 - Leaky ReLU
    x = LeakyReLU(alpha = RELU_SLOPE_C, name='Activation_4')(x)

    # Dropout 4
    x = Dropout(rate = DROPOUT_C)(x)




    # Conv2D Layer 5
    x = Conv2D(filters = 512,
               kernel_size = 5,
               strides = 2,
               padding = 'same',
               name = 'Discriminator_Conv2D_Layer_5',
               kernel_initializer = RandomNormal(mean=0., stddev=0.02)
    )(x)

    # BatchNorm Layer 5
    # x = BatchNormalization()(x)

    # Activation 5 - Leaky ReLU
    x = LeakyReLU(alpha = RELU_SLOPE_C, name='Activation_5')(x)

    # Dropout 5
    #x = Dropout(rate = 0.4)(x)


    # OUTPUT
    x = Flatten()(x)
    x = Dropout(DROPOUT_C)(x)
    d_output = Dense(1, 
                     #activation='sigmoid', 
                     kernel_initializer = RandomNormal(mean=0, stddev=0.02) # random initialization of weights with normal distribution around 0 with small SD
                    )(x)



    # Discriminator Model intialization
    discriminator = Model(d_input, d_output, name='Discriminator')
    
    
    
    return discriminator


# In[12]:

'''
disc = discriminator_model()
disc.summary()
'''

# ### Generator

# In[9]:


def generator_model():

    # GENERATOR

    # set input variable dimensions. Here we will start out with a vector of length 100 for each sample (sampled from a normal distribution, representing the learned latent space)
    input_dim = (LATENT_DIM)

    # Input
    g_input = Input(shape=input_dim, name='Generator_Input')

    # Dense Layer 1
    x = Dense(np.prod([4,4,2048]), kernel_initializer = RandomNormal(mean=0., stddev=0.02), 
              use_bias=False)(g_input) # use_bias=False see https://keras.io/examples/generative/wgan_gp/

    # Batch Norm Layer 1
    x = BatchNormalization(momentum = MOMENTUM_G)(x)

    # Activation Layer 1
    x = LeakyReLU(alpha=RELU_SLOPE_G)(x) # trying leaky ReLU instead of Activation('relu')(x)

    # Reshape into 3D tensor
    x = Reshape((4,4,2048))(x)

    # Upsampling Layer 1 + Conv2D Layer1
    x = Conv2DTranspose(filters=1024, kernel_size=5, padding='same', strides=(2,2), 
                        kernel_initializer = RandomNormal(mean=0., stddev=0.02), use_bias=False)(x)
    
    
    # Batch Norm Layer 2
    x = BatchNormalization(momentum = MOMENTUM_G)(x)
    
    # Activation Layer 2
    x = LeakyReLU(alpha=RELU_SLOPE_G)(x) # trying leaky ReLU instead of Activation('relu')(x)
    
    #x = UpSampling2D()(x)
    #x = Conv2D(filters=64, kernel_size=5, padding='same', name='Generator_Conv_Layer_1')(x)
    # Replacing by Conv2DTranspose Layer
    x = Conv2DTranspose(filters=512, kernel_size=5, padding='same', strides=(2,2), 
                        kernel_initializer = RandomNormal(mean=0., stddev=0.02), use_bias=False)(x)
    
    
    # Batch Norm Layer 2
    x = BatchNormalization(momentum = MOMENTUM_G)(x)

    # Activation Layer 2
    x = LeakyReLU(alpha=RELU_SLOPE_G)(x) # trying leaky ReLU instead of Activation('relu')(x)

    # Upsampling Layer 2 + Conv2D Layer2
    #x = UpSampling2D()(x)
    #x = Conv2D(filters=128, kernel_size=5, padding='same', name='Generator_Conv_Layer_2')(x)
    # Replacing by Conv2DTranspose Layer
    x = Conv2DTranspose(filters=256, kernel_size=5, padding='same', strides=(2,2), 
                        kernel_initializer = RandomNormal(mean=0., stddev=0.02), use_bias=False)(x)

    # Batch Norm Layer 3
    x = BatchNormalization(momentum = MOMENTUM_G)(x)

    # Activation Layer 3
    x = LeakyReLU(alpha=RELU_SLOPE_G)(x) # trying leaky ReLU instead of Activation('relu')(x)

    # Upsampling Layer 3 + Conv2D Layer3
    #x = UpSampling2D()(x)
    #x = Conv2D(filters=64, kernel_size=5, padding='same', name='Generator_Conv_Layer_3')(x)
    # Replacing by Conv2DTranspose Layer
    x = Conv2DTranspose(filters=128, kernel_size=5, padding='same', strides=(2,2), 
                        kernel_initializer = RandomNormal(mean=0., stddev=0.02), use_bias=False)(x)

    
    #### 
    # START
    # SKIP-Z BLOCK 1
    ####
    
    if SKIP:
        skip = Dense(np.prod([64, 64, 128]))(g_input)
        skip = Reshape((64, 64, 128))(skip)

        x = Add()([x, skip])
    
    #### 
    # END
    # SKIP-Z BLOCK 1
    ####
    
    
    
    # Batch Norm Layer 4
    x = BatchNormalization(momentum = MOMENTUM_G)(x)

    # Activation Layer 4
    x = LeakyReLU(alpha=RELU_SLOPE_G)(x) # trying leaky ReLU instead of Activation('relu')(x)

    # Upsampling Layer 4 + Conv2D Layer4
    #x = UpSampling2D()(x)
    #x = Conv2D(filters=64, kernel_size=5, padding='same', name='Generator_Conv_Layer_4')(x)
    # Replacing by Conv2DTranspose Layer
    
    #removed for 128x128 pictures
   
    x = Conv2DTranspose(filters=64, kernel_size=5, padding='same', strides=1, 
                        kernel_initializer = RandomNormal(mean=0., stddev=0.02), use_bias=False)(x)
    
    #### 
    # START
    # SKIP-Z BLOCK 2
    ####
    
    if SKIP:
        skip = Dense(np.prod([64, 64, 64]))(g_input)
        skip = Reshape((64, 64, 64))(skip)

        x = Add()([x, skip])
    
    #### 
    # END
    # SKIP-Z BLOCK 2
    ####
    

    # Batch Norm Layer 5
    x = BatchNormalization(momentum = MOMENTUM_G)(x)
    

    # Activation Layer 5
    x = LeakyReLU(alpha=RELU_SLOPE_G)(x) # trying leaky ReLU instead of Activation('relu')(x)
    
    
    '''
    x = Conv2DTranspose(filters=3, kernel_size=7, padding='same', strides=(2,2), 
                        kernel_initializer = RandomNormal(mean=0., stddev=0.02), use_bias=False)(x)
                        
                        
    x = Conv2DTranspose(filters=3, kernel_size=(1,1), strides=(12,10), 
                        kernel_initializer = RandomNormal(mean=0., stddev=0.02), use_bias=False)(x)
    '''
    
    # non-symmetric kernel with non-overlapping stride?
    x = Conv2DTranspose(filters=3, kernel_size=3, padding='same', strides=(2,2), 
                        kernel_initializer = RandomNormal(mean=0., stddev=0.02), use_bias=False)(x)
    
    
    
    
    #### 
    # START
    # SKIP-Z BLOCK 3
    ####
    
    if SKIP:
        skip = Dense(np.prod([128, 128, 3]))(g_input)
        skip = Reshape((128, 128, 3))(skip)

        x = Add()([x, skip])
    
    #### 
    # END
    # SKIP-Z BLOCK 3
    ####
   

    # Activation Layer 7
    #x = LeakyReLU()(x) # trying leaky ReLU instead of Activation('relu')(x)


    # Upsampling Layer 8 + Conv2D Layer8
    #x = Conv2D(filters=3, kernel_size=5, padding='same', name='Generator_Conv_Layer_7')(x)
    # ---------------------------/
    
    # tanh activation layer to scale values to [-1:1]
    x = Activation('tanh')(x)
    
    # Batch Norm Layer 7
    #x = BatchNormalization(momentum = MOMENTUM_G)(x)
    x = LayerNormalization()(x) # performs pixel-wise normalization across all channels

    # Output - Rescale Values back to [0:255] since the discriminator will automatically rescale back down to [-1:1] as part of the pre-processing pipeline
    g_output = (255 / 2) * (x + 1) 


    # Generator Model initialization
    generator = Model(g_input, g_output, name='Generator')
    
    
    return generator


# In[15]:

'''
gen = generator_model()
gen.summary()
'''

# ### WGAN-GP (Full) Model Compilation

# Before we compile the models, we need to implement a custom loss function which uses the Wasserstein distance and a gradient penalty term in order to ensure 1 Lipschitz constraints are followed. A WGAN with GP further involves a slightly more complicated training process which trains the critic (discriminator without sigmoid activation function) by feeding three different kinds of images:
# 
# 1. real images (i.e. available samples)
# 2. 'fake' images (i.e. constructed by the generator)
# 3. random interpolations between real and fake images (i.e. random samples interpolated from values between the fake and real images)
# 
# The full training process of a critic is depicted below (source: Foster, 2019, p. 122):
# 
# <div align="center">
#     <img src="wgan_gp_critic_training.png"></img>
#     <i>Computational Graph for one Discriminator Training Epoch. (Source: Foster, 2019, p.122)</i>
# </div>

# The implementation below roughly follows the OOP-based framework set by keras see https://keras.io/examples/generative/wgan_gp/

# In[10]:


strategy = tf.distribute.MirroredStrategy()

with strategy.scope(): 

    critic = discriminator_model()
    generator = generator_model()

    class WGANGP(keras.Model):
        def __init__(
            self,
            critic,
            generator,
            latent_dim,
            tensorboard_callback,
            critic_extra_steps=5,
            gp_weight=10.0
        ):
            super(WGANGP, self).__init__()
            self.critic = critic
            self.generator = generator
            self.latent_dim = latent_dim
            self.tensorboard_callback = tensorboard_callback
            self.d_steps = critic_extra_steps
            self.gp_weight = gp_weight
            

        def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
            super(WGANGP, self).compile()
            self.d_optimizer = d_optimizer
            self.g_optimizer = g_optimizer
            self.d_loss_fn = d_loss_fn
            self.g_loss_fn = g_loss_fn

        def gradient_penalty(self, batch_size, real_images, fake_images):
            """ Calculates the gradient penalty.

            This loss is calculated on an interpolated image
            and added to the discriminator loss.
            """
            # Get the interpolated image
            alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
            diff = fake_images - real_images
            interpolated = real_images + alpha * diff

            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                # 1. Get the discriminator output for this interpolated image.
                pred = self.critic(interpolated, training=True)

            # 2. Calculate the gradients w.r.t to this interpolated image.
            grads = gp_tape.gradient(pred, [interpolated])[0]
            # 3. Calculate the norm of the gradients.
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
            gp = tf.reduce_mean((norm - 1.0) ** 2)
            return gp

        def train_step(self, real_images):
            #checking whether we handed a tuple of (numpy) data to .fit().
            #if not, the data must be a tf.data.Dataset generator that yields batches of datasets (data, labels)
            if isinstance(real_images, tuple):
                real_images = real_images[0]

            # Get the batch size
            batch_size = tf.shape(real_images)[0]

            # For each batch, we are going to perform the
            # following steps as laid out in the original paper:
            # 1. Train the generator and get the generator loss
            # 2. Train the discriminator and get the discriminator loss
            # 3. Calculate the gradient penalty
            # 4. Multiply this gradient penalty with a constant weight factor = self.discriminator_extra_steps = 5 (default value)
            # 5. Add the gradient penalty to the discriminator loss
            # 6. Return the generator and discriminator losses as a loss dictionary

            # Train the discriminator first. The original paper recommends training
            # the discriminator for `x` more steps (typically 5) as compared to generator
            
            # set tensorboard model
            self.tensorboard_callback.set_model(self.critic)
            
            for i in range(self.d_steps):
                # Get the latent vector
                random_latent_vectors = tf.random.normal(
                    shape=(batch_size, self.latent_dim)
                )
                with tf.GradientTape() as tape:
                    # Generate fake images from the latent vector
                    fake_images = self.generator(random_latent_vectors, training=True)
                    # Get the logits for the fake images
                    fake_logits = self.critic(fake_images, training=True)
                    # Get the logits for the real images
                    real_logits = self.critic(real_images, training=True)

                    # Calculate the discriminator loss using the fake and real image logits
                    d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                    # Calculate the gradient penalty
                    gp = self.gradient_penalty(batch_size, real_images, fake_images)
                    # Add the gradient penalty to the original discriminator loss
                    d_loss = d_cost + gp * self.gp_weight

                # Get the gradients w.r.t the discriminator loss
                d_gradient = tape.gradient(d_loss, self.critic.trainable_variables)
                # Update the weights of the discriminator using the discriminator optimizer
                self.d_optimizer.apply_gradients(
                    zip(d_gradient, self.critic.trainable_variables)
                )

            # Train the generator
            # set tensorboard model
            self.tensorboard_callback.set_model(self.generator)
            
            # Get the latent vector
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                # Generate fake images using the generator
                generated_images = self.generator(random_latent_vectors, training=True)
                # Get the discriminator logits for fake images
                gen_img_logits = self.critic(generated_images, training=True)
                # Calculate the generator loss
                g_loss = self.g_loss_fn(gen_img_logits)

            # Get the gradients w.r.t the generator loss
            gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
            # Update the weights of the generator using the generator optimizer
            self.g_optimizer.apply_gradients(
                zip(gen_gradient, self.generator.trainable_variables)
            )
            return {"d_loss": d_loss, "g_loss": g_loss}
        
    class GANMonitor(keras.callbacks.Callback):
        def __init__(self, num_img=5, latent_dim=128):
            self.num_img = num_img
            self.latent_dim = latent_dim

        def on_epoch_end(self, epoch, logs=None): #on_epoch_end(self, epoch, logs=None):
            '''
            random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
            generated_images = self.model.generator(random_latent_vectors)
            #generated_images = (generated_images * 127.5) + 127.5

            for i in range(self.num_img):
                img = generated_images[i].numpy()
                img = keras.preprocessing.image.array_to_img(img)
                img.save("generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))
            '''
            
            # Sample generator output for num_img images
            noise = np.random.normal(0, 1, (self.num_img, self.latent_dim))
            gen_imgs = generator.predict(noise)

            #!!!NOT NECESSARY ANYMORE AS IMPLEMENTED AS PART OF THE MODEL!!!
            #gen_imgs = 0.5 * (gen_imgs + 1)  #scale back to [0:1]
            gen_imgs = gen_imgs.reshape((self.num_img, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

            # save n example images
            for i in range(self.num_img):
                fig = plt.figure(figsize=(10, 10))
                plt.imshow(gen_imgs[i].astype('uint8'))
                plt.title(f'Example Generator Output')
                plt.axis('off')

                # adjust path based on whether execution is local or on linux VM
                if pathlib.Path(f'{out_img_dir}/{model_name}').exists():
                    fig.savefig(f'{out_img_dir}/{model_name}/sample_image_epoch{epoch+1}-{i+1}.png')
                    plt.close()
                else:
                    #mkdir
                    os.mkdir(f'{out_img_dir}/{model_name}')
                    #save
                    fig.savefig(f'{out_img_dir}/{model_name}/sample_image_epoch{epoch+1}-{i+1}.png')
                    plt.close()
                    
            # save corresponding model
            now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
            
            if pathlib.Path(f'{out_model_dir}/{model_name}').exists():
                #gan.save(f'{out_model_dir}/{model_name}/full-gan-{now}.h5')
                critic.save(f'{out_model_dir}/{model_name}/critic-{now}.h5') 
                generator.save(f'{out_model_dir}/{model_name}/generator-{now}.h5')        
            else:
                #make dir
                os.mkdir(f'{out_model_dir}/{model_name}')
                #write
                #gan.save(f'{out_model_dir}/{model_name}/full-gan-{now}.h5')
                critic.save(f'{out_model_dir}/{model_name}/critic-{now}.h5') 
                generator.save(f'{out_model_dir}/{model_name}/generator-{now}.h5') 
        
        
    # Instantiate the optimizer for both networks
    # (learning_rate=0.0002, beta_1=0.5 are recommended) as per Radford et al. 2016 pp. 3-4
    generator_optimizer = Adam(
        learning_rate=GEN_LR, beta_1=0.5, beta_2=0.9
    )
    critic_optimizer = Adam(
        learning_rate=CRIT_LR, beta_1=0.5, beta_2=0.9
    )

    # Define the loss functions for the discriminator,
    # which should be (fake_loss - real_loss).
    # We will add the gradient penalty later to this loss function.
    def critic_loss(real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss


    # Define the loss functions for the generator.
    def generator_loss(fake_img):
        return -tf.reduce_mean(fake_img)


    # Instantiate the custome `GANMonitor` Keras callback.
    cbk = GANMonitor(num_img=5, latent_dim=LATENT_DIM)
    
    # Instantiate the tensorboard tf.keras callback
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')

    tb_cbk = tf.keras.callbacks.TensorBoard(
        log_dir = f'{tboard_dir}/{model_name}_{now}', 
        write_graph = False,
        write_images = True,
        histogram_freq = 1) 

    # Instantiate the WGAN model.
    wgan = WGANGP(
        critic=critic,
        generator=generator,
        latent_dim=LATENT_DIM,
        tensorboard_callback=tb_cbk,
        critic_extra_steps=CRITIC_FACTOR,
        gp_weight=GRADIENT_PENALTY_WEIGHT
        
    )

    # Compile the WGAN model.
    wgan.compile(
        d_optimizer=critic_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=critic_loss,
    )


# In[11]:


# Start training the model.
wgan.fit(train_ds, batch_size=BATCH_SIZE, epochs=200, callbacks=[cbk, tb_cbk])


# ### Generate Examples using learned Generator Model

# #### After 100 Epochs

# In[67]:
'''

# load generator
'''
generator.compile(optimizer=Adam(lr=0.0008), # per Foster, 2017 RMSprop(lr=0.0008)
                          loss=binary_crossentropy,
                          metrics=['accuracy'])
generator = tf.keras.models.load_model('/data/output/models/dwarfganWGANGPR02/generator-2021-04-04_025322.h5')
'''
# generate new example of learned representation in latent space
try:
    generator
except NameError:
    #get latest generator model save file
    folder = pathlib.Path(f'{out_model_dir}/{model_name}')
    saves = list(folder.glob('generator*'))
    latest = max(saves, key=os.path.getctime)
    #load latest generator save file
    generator = tf.keras.models.load_model(latest)
        
noise = np.random.normal(0, 1, (1, LATENT_DIM))
res = np.array(generator(noise, training=False)).astype('uint8')

#Rescale
res = res.reshape((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

# Visualize result
plt.figure(figsize=(8, 8))
plt.imshow(res)
plt.title(f'Example Generator Output')
plt.axis('off')


# # TODOs

# ### Changes made after 10 epoche run (V0.0)
# ---
# - <img src="https://cdn1.iconfinder.com/data/icons/warnings-and-dangers/400/Warning-02-512.png" width=20></img> kernel initializer in discriminator Conv2D layers set to RandomNormal(mean=0., stddev=0.02) (before default value = glorot_uniform aka Xavier)
# 
# ### Changes to be made after 100 epoche run (for V0.5)
# ---
# - Batch size 32 --> 64. Potentially use increasing batch size (rather than decay of learning rate) see: https://stackoverflow.com/questions/50606995/how-to-change-the-batch-size-during-training  --> might lead to memory problems in current 44GB RAM VM
# - Dynamically increase batch size instead of decaying learning rate to accelarate training see https://arxiv.org/abs/1711.00489
# - check some filters on different Conv2D layers --> use smaller filter to detect relatively small features?
# - adjust filter size for some layers (5x5 -> 10x10?) --> shrinking size too fast is not helpful. maybe add additional conv layers to shrink more slowely?
# - <img src="https://cdn1.iconfinder.com/data/icons/warnings-and-dangers/400/Warning-02-512.png" width=20></img> Discriminator Optimizer Adam -> RMSprop
# - <img src="https://cdn1.iconfinder.com/data/icons/warnings-and-dangers/400/Warning-02-512.png" width=20></img> Added discriminator.trainable = True/False flags to training functions for discriminator/generator (GAN) to ensure discriminator is actually trainable
# - <img src="https://cdn1.iconfinder.com/data/icons/warnings-and-dangers/400/Warning-02-512.png" width=20></img> So far only 3652 images were used for training (single image maps and map archives starting with X, Y or Z). The next training cycle will include maps starting with U, V or W --> added all available samples
# - <img src="https://cdn1.iconfinder.com/data/icons/warnings-and-dangers/400/Warning-14-512.png" width=20></img> add 'sample_img()' function to execute every 'sample_interval' - added but not tested
# - <img src="https://cdn1.iconfinder.com/data/icons/warnings-and-dangers/400/Warning-14-512.png" width=20></img> tune polarization method in input data pipeline which scales pictures to 1024 x 1024? OR randomly crop a 1024 x 1024 piece of the input image to better preserve size of objects rather than scaling everything to 1024x1024 ==> random crop changed to 512x512 due to OOM error in training on GPUs
# ==> pre-processing applied to input 100k input images of different size: random cropped 10 1024x1024 regions and dropping all black etc. images, generating over 700k cropped samples
# - <img src="https://cdn1.iconfinder.com/data/icons/warnings-and-dangers/400/Warning-02-512.png" width=20></img> implement tensorboard for model monitoring during training
# - try zero centering (mean subtracting) input data?
# - use Randomized ReLU instead of leaky see: https://arxiv.org/pdf/1505.00853.pdf
# - add another loop in train_gan() over different steps within training loop?
# - <img src="https://cdn1.iconfinder.com/data/icons/warnings-and-dangers/400/Warning-02-512.png" width=20></img> add slight randomness to labels (see Chollet p. 307)
# - implement wasserstein GAN with gradient penalty (WGAN-GP, see  Foster p. 115 ff)
# - use Frechet Inception Distance (FIP) for evaluation of GAN performance (see https://arxiv.org/abs/1706.08500 )

# # Notes - Besprechung Vorstudie 15.01.2021
# 
# - Ausdrücke aus der Game Welt evtl. näher erläutern erläutern bspw. "Game Assets"
# - Spezifizierung Forschungsfragen --> Ziele etwas zu offen bspw. was heisst "sufficiently able to learn" --> include Frechet Inception Distance (FIP)
#     -- konkretisieren
#     -- 2 Milestones (konkret) formulieren für Ergebnis mit Termin pro Milestones für Vorstellung dieser Milestones (mit Thomas Koller im CC inkl. Vorstudie) --> Für Herrn Birbaumer am liebsten <b>Donnerstag Nachmittag</b>:
#         --- Milestone 1: Frechet Distance für ersten milestone 1 + WGAN-GP? Einführung nochmals Vorstudie als überblick
#         --- Milestone 2: ???
# - Bewertung von Games / Vergleichsstudie --> wie können Game Devs dazu bewogen werden diese zu verwenden, was sind das für Kriterien
# - 2-3 Wochen vor Abgabe "Abschlusspresentation" einplanen
'''
