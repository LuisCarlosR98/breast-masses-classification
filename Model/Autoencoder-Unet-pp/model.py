import tensorflow as tf
from tensorflow.keras import layers, Model, Input

def conv_block(x, filters, name):
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=name+'_conv1')(x)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=name+'_conv2')(x)
    return x

def unet_plus_plus(input_shape=(256, 256, 1), num_classes=1):
    inputs = Input(shape=input_shape)

    # Encoder
    x00 = conv_block(inputs, 64, 'x00')
    x10 = layers.MaxPooling2D((2, 2))(x00)
    x10 = conv_block(x10, 128, 'x10')

    x20 = layers.MaxPooling2D((2, 2))(x10)
    x20 = conv_block(x20, 256, 'x20')

    x30 = layers.MaxPooling2D((2, 2))(x20)
    x30 = conv_block(x30, 512, 'x30')

    x40 = layers.MaxPooling2D((2, 2))(x30)
    x40 = conv_block(x40, 1024, 'x40')

    # Decoder
    x01 = conv_block(layers.Concatenate()([x00, layers.UpSampling2D((2, 2), interpolation='bilinear')(x10)]), 64, 'x01')
    x11 = conv_block(layers.Concatenate()([x10, layers.UpSampling2D((2, 2), interpolation='bilinear')(x20)]), 128, 'x11')
    x21 = conv_block(layers.Concatenate()([x20, layers.UpSampling2D((2, 2), interpolation='bilinear')(x30)]), 256, 'x21')
    x31 = conv_block(layers.Concatenate()([x30, layers.UpSampling2D((2, 2), interpolation='bilinear')(x40)]), 512, 'x31')

    x02 = conv_block(layers.Concatenate()([x00, x01, layers.UpSampling2D((2, 2), interpolation='bilinear')(x11)]), 64, 'x02')
    x12 = conv_block(layers.Concatenate()([x10, x11, layers.UpSampling2D((2, 2), interpolation='bilinear')(x21)]), 128, 'x12')
    x22 = conv_block(layers.Concatenate()([x20, x21, layers.UpSampling2D((2, 2), interpolation='bilinear')(x31)]), 256, 'x22')

    x03 = conv_block(layers.Concatenate()([x00, x01, x02, layers.UpSampling2D((2, 2), interpolation='bilinear')(x12)]), 64, 'x03')
    x13 = conv_block(layers.Concatenate()([x10, x11, x12, layers.UpSampling2D((2, 2), interpolation='bilinear')(x22)]), 128, 'x13')

    x04 = conv_block(layers.Concatenate()([x00, x01, x02, x03, layers.UpSampling2D((2, 2), interpolation='bilinear')(x13)]), 64, 'x04')

    output = layers.Conv2D(num_classes, (1, 1), activation='sigmoid', name='final_output')(x04)
    model = Model(inputs, output)
    return model
