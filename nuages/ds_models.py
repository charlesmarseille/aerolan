from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Concatenate
from keras.layers import UpSampling2D
from keras.layers import MaxPooling2D
from keras.layers import Conv2DTranspose
from keras.layers import Add
from keras.layers import Multiply
from keras.layers import Subtract
from keras.layers import Lambda
from keras.layers import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.initializers import RandomNormal
import numpy as np

def encoder_block(layer_in, n_filters, filter_size, activation):
    init = RandomNormal(stddev=0.02)
    c1_1 = Conv2D(n_filters, filter_size, activation=activation, padding='same', dilation_rate=1, kernel_initializer=init)(layer_in)
    c1_2 = Conv2D(n_filters, filter_size, activation=activation, padding='same', dilation_rate=2, kernel_initializer=init)(c1_1)
    c1 = Add()([c1_1, c1_2])
    c1 = BatchNormalization()(c1)

    c2_1 = Conv2D(n_filters, filter_size, activation=activation, padding='same', dilation_rate=1, kernel_initializer=init)(c1)
    c2_2 = Conv2D(n_filters, filter_size, activation=activation, padding='same', dilation_rate=2, kernel_initializer=init)(c2_1)
    c_skip = Add()([c2_1, c2_2])
    c_skip = BatchNormalization()(c_skip)
    # dropout
    c = MaxPooling2D(pool_size=(2,2), padding='same')(c_skip)
    return c, c_skip

def bottleneck_block(layer_in, n_filters, filter_size, activation):
    init = RandomNormal(stddev=0.02)
    c1 = Conv2D(n_filters, filter_size, activation=activation, padding='same', dilation_rate=1, kernel_initializer=init)(layer_in)
    c2 = Conv2D(n_filters, filter_size, activation=activation, padding='same', dilation_rate=2, kernel_initializer=init)(c1)
    c3 = Conv2D(n_filters, filter_size, activation=activation, padding='same', dilation_rate=4, kernel_initializer=init)(c2)
    c4 = Conv2D(n_filters, filter_size, activation=activation, padding='same', dilation_rate=8, kernel_initializer=init)(c3)
    #c5 = Conv2D(n_filters, filter_size, activation=activation, padding='same', dilation_rate=16, kernel_initializer=init)(c4)
    #c6 = Conv2D(n_filters, filter_size, activation=activation, padding='same', dilation_rate=32, kernel_initializer=init)(c5)

    c = Add()([c1, c2, c3, c4])#, c5, c6])
    c = Dropout(0.2)(c, training=True)
    c = Activation(activation)(c)
    c = BatchNormalization()(c)
    return c

def decoder_block(layer_in, skip_in, n_filters, filter_size, activation, dropout=True):
    init = RandomNormal(stddev=0.02)
    skip_in = Conv2D(n_filters, filter_size, activation=activation, padding='same', dilation_rate=1, kernel_initializer=init)(skip_in)
    c = Conv2D(n_filters, filter_size, activation=activation, padding='same', kernel_initializer=init)(layer_in)
    c = UpSampling2D(size=(2, 2))(c)
    c = Concatenate()([c, skip_in])

    if dropout:
        c = Dropout(0.2)(c, training=True)
    c = BatchNormalization()(c)
    c = Activation(activation)(c)
    return c

def define_unet_discriminator(config):
    # Read configuration
    init = RandomNormal(stddev=0.02)
    n_filters = config['n_filters']
    filter_size = config['filter_size']
    activation = config['activation']
    n_channels_in = config['n_channels_in']
    img_size = config['img_size']
    loss = config['loss']
    loss_weights = config['loss_weights']
    lr = config['lr']

    # Define attention block
    filters_mult = [1, 2, 4, 4, 4, 2, 1]
    in_image = Input(shape=(img_size[0], img_size[1], n_channels_in))
    c1, c1_skip = encoder_block(in_image, filters_mult[0]*n_filters, filter_size, activation)
    c2, c2_skip = encoder_block(c1, filters_mult[1]*n_filters, filter_size, activation)
    c3, c3_skip = encoder_block(c2, filters_mult[2]*n_filters, filter_size, activation)
    c4 = bottleneck_block(c3, filters_mult[3]*n_filters, filter_size, activation)
    c5 = decoder_block(c4, c3_skip, filters_mult[4]*n_filters, filter_size, activation)
    c6 = decoder_block(c5, c2_skip, filters_mult[5]*n_filters, filter_size, activation, dropout=False)
    c7 = decoder_block(c6, c1_skip, filters_mult[6]*n_filters, filter_size, activation, dropout=False)
    patch_out = Conv2D(int(n_filters/2), (1, 1), activation=activation)(c7)
    patch_out = Conv2D(1, (1, 1))(patch_out)

    # compile model
    model = Model(in_image, patch_out)
    model.compile(loss=loss, optimizer=Adam(lr=lr, beta_1=0.5, epsilon=0.00005), loss_weights=loss_weights)
    return model

def define_aNet(config):
    # Read configuration
    init = RandomNormal(stddev=0.02)
    n_filters = config['n_filters']
    filter_size = config['filter_size']
    activation = config['activation']
    c_activation = config['c_activation']
    n_channels_in = config['n_channels_in']
    n_channels_out = config['n_channels_out']
    img_size = config['img_size']
    
    if activation == 'lrelu':
        activation = LeakyReLU(alpha=0.3)

    # Prepare input
    in_image = Input(shape=(img_size[0], img_size[1], n_channels_in))

    # Define encoder block
    filters_mult = [1, 2, 4, 4, 4, 2, 1]
    c1, c1_skip = encoder_block(in_image, filters_mult[0]*n_filters, filter_size, activation)
    c2, c2_skip = encoder_block(c1, filters_mult[1]*n_filters, filter_size, activation)
    c3, c3_skip = encoder_block(c2, filters_mult[2]*n_filters, filter_size, activation)
    c4 = bottleneck_block(c3, filters_mult[3]*n_filters, filter_size, activation)

    # Define attention decoder block
    c5 = decoder_block(c4, c3_skip, filters_mult[4]*n_filters, filter_size, activation)
    c6 = decoder_block(c5, c2_skip, filters_mult[5]*n_filters, filter_size, activation, dropout=False)
    c7 = decoder_block(c6, c1_skip, filters_mult[6]*n_filters, filter_size, activation, dropout=False)
    out_attn = Conv2D(int(n_filters/2), (1, 1), activation=activation)(c7)
    out_attn = Conv2D(1, (1, 1), activation='sigmoid')(out_attn)

    # Define generator decoder block
    c5 = decoder_block(c4, c3_skip, filters_mult[4]*n_filters, filter_size, activation)
    c6 = decoder_block(c5, c2_skip, filters_mult[5]*n_filters, filter_size, activation, dropout=False)
    c7 = Conv2D(n_filters, filter_size, activation=activation, padding='same', kernel_initializer=init)(c6)
    c7 = UpSampling2D(size=(2, 2))(c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation(activation)(c7)
    out_gen = Conv2D(int(n_filters/2), (1, 1), activation=activation)(c7)
    out_gen = Conv2D(n_channels_out, (1, 1), activation=c_activation)(out_gen)

    # Fuse predictions
    out_1 = Multiply()([out_attn, out_gen])
    #inv_attn = 1 - out_attn
    inv_attn = Lambda(lambda x: 1. - x)(out_attn)
    out_2 = Multiply()([inv_attn, in_image])
    out_image = Add()([out_1, out_2])

    # Compile model
    model = Model(in_image, [out_image, out_gen, out_attn])
    return model

def define_my_composite_model(g_model, d_model, config):
    # Read configuration
    n_channels_in = config['n_channels_in']
    img_size = config['img_size']
    loss = config['loss']
    loss_weights = config['loss_weights']
    lr = config['lr']

    # mark discriminator as not trainable
    d_model.trainable = False
    # ensure the model we're updating is trainable
    g_model.trainable = True

    # BtoA element
    input_gen = Input(shape=(img_size[0], img_size[1], n_channels_in))
    gen_out, unet_out, _ = g_model(input_gen)
    d_out = d_model(gen_out)

    # AtoA element
    input_id = Input(shape=(img_size[0], img_size[1], n_channels_in))
    gen_id_out, unet_id_out, attn_id_out = g_model(input_id)
    d_id_out = d_model(gen_id_out)

    # define model
    model = Model([input_gen, input_id], [gen_out, unet_out, d_out, gen_id_out, attn_id_out, unet_id_out])
    model.compile(loss=loss, loss_weights=loss_weights, optimizer=Adam(lr=lr, beta_1=0.5))
    return model
