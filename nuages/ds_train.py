import sys
import os
from os import listdir
import numpy as np
import cv2
from matplotlib import pyplot
from ds_utils import get_training_data, load_paths, summarize_performance, update_image_pool
from ds_models import define_my_composite_model, define_aNet, define_unet_discriminator

# Define paths
load_path = 'C:/Users/15793/Documents/Projets/nuages/data/'
save_path = 'C:/Users/15793/Documents/Projets/nuages/results/'

# Prepare models config
IMG_SIZE = 64
discriminator_config = {
    'img_size': (IMG_SIZE, IMG_SIZE),
    'n_channels_in': 3,
    'n_filters': 32,
    'filter_size': (3,3),
    'activation': 'relu',
    'lr': 0.00003,
    'loss': 'mae',
    'loss_weights': [0.5],
    }
generator_config = {
    'img_size': (IMG_SIZE, IMG_SIZE),
    'n_channels_in': 3,
    'n_channels_out': 3,
    'n_filters': 64,
    'filter_size': (3,3),
    'activation': 'lrelu',
    'c_activation': 'tanh',
    'opt': 'Adam',
    'lr': 0.00005,
    'loss': ['mae', 'mae', 'mse', 'mae', 'mae', 'mae'],
    'loss_weights': [1, 2, 1, 3, 4, 1],
    }

# Define data augmentation parameters
aug_params = {
    'img_size': (IMG_SIZE, IMG_SIZE),
    'spatial_aug_ratio': 1.0,
    'info_aug_ratio': 0.0,
    }

# Define Models
d_model = define_unet_discriminator(discriminator_config)
g_model = define_aNet(generator_config)
c_model = define_my_composite_model(g_model, d_model, generator_config)
#c_model.summary()
# Load dataset and patches for adversarial training
trainA, trainB = load_paths(load_path)

# Train model
n_epochs = 200
n_batch = 1
n_to_save = 100
steps_per_epoch = len(trainA)
true_patch = np.ones((n_batch, IMG_SIZE, IMG_SIZE, 1))
false_patch = np.zeros((n_batch, IMG_SIZE, IMG_SIZE, 1))
poolA, poolB = list(), list()
dg1_losses_, dg2_losses_, g_losses_ = [], [], []

for i in range(n_epochs * steps_per_epoch):
    # select a batch of real samples
    X_realA = get_training_data(trainA, aug_params, n_batch)
    X_realB = get_training_data(trainB, aug_params, n_batch)
    #print(np.min(X_realB[0]), np.max(X_realB[0]))
    #pyplot.imshow(((X_realB[0][:,:,:]+1)*127.5).astype(np.uint8))
    #pyplot.show()
    
    # generate a batch of fake samples
    #X_fakeA, _, _ = g_model.predict(X_realA)
    X_fakeB, _, _ = g_model.predict(X_realB)
    # update pools
    #X_fakeA = update_image_pool(poolA, X_fakeA)
    X_fakeB = update_image_pool(poolB, X_fakeB)

    # update discriminator for "real or generated"
    dg_loss1 = d_model.train_on_batch(X_realA, true_patch)
    dg_loss2 = d_model.train_on_batch(X_fakeB, false_patch)
    # update generator A->B via adversarial and cycle loss
    g_loss, _, _, _, _, _, _ = c_model.train_on_batch([X_realB, X_realA], [X_realB, X_realB, true_patch, X_realA, false_patch, X_realA])

    # summarize performance
    dg1_losses_.append(dg_loss1)
    dg2_losses_.append(dg_loss2)
    g_losses_.append(g_loss)
    if i % steps_per_epoch == 0:
        print('epoch %d: dg[%.3f,%.3f] g[%.3f]' % (i//steps_per_epoch+1, np.mean(dg1_losses_),np.mean(dg2_losses_), np.mean(g_losses_)))
        dg1_losses_ = []
        dg2_losses_ = []
        g_losses_ = []

    # Make prediction
    if (i+1) % (steps_per_epoch * n_epochs / n_to_save) == 0:
        print('Starting validation test...')
        summarize_performance(i, g_model, trainB, aug_params, save_path, i)
