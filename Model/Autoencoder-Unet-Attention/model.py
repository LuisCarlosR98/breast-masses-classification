import tensorflow as tf
from tensorflow.keras import layers, models

# Módulo de Atención Cruzada Dual (DCA)
def dual_cross_attention(encoder_feature, decoder_feature, filters):
    # Reducción de dimensiones
    encoder_proj = layers.Conv2D(filters, (1, 1), padding='same')(encoder_feature)
    decoder_proj = layers.Conv2D(filters, (1, 1), padding='same')(decoder_feature)

    # Channel Cross-Attention (CCA)
    encoder_gap = layers.GlobalAveragePooling2D()(encoder_proj)
    decoder_gap = layers.GlobalAveragePooling2D()(decoder_proj)
    channel_mul = layers.Multiply()([encoder_gap, decoder_gap])
    channel_attention = layers.Dense(filters, activation='sigmoid')(channel_mul)
    channel_attention = layers.Reshape((1, 1, filters))(channel_attention)
    encoder_channel_att = layers.Multiply()([encoder_proj, channel_attention])

    # Spatial Cross-Attention (SCA)
    encoder_conv = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(encoder_proj)
    decoder_conv = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(decoder_proj)
    spatial_concat = layers.Concatenate()([encoder_conv, decoder_conv])
    spatial_attention = layers.Conv2D(1, (1, 1), activation='sigmoid')(spatial_concat)
    encoder_spatial_att = layers.Multiply()([encoder_channel_att, spatial_attention])

    return encoder_spatial_att

# Arquitectura U-Net con DCA
def unet_autoencoder_dca(input_shape=(256, 256, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder con DCA
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    c4_att = dual_cross_attention(c4, u6, 512)
    u6 = layers.Concatenate()([u6, c4_att])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    c3_att = dual_cross_attention(c3, u7, 256)
    u7 = layers.Concatenate()([u7, c3_att])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    c2_att = dual_cross_attention(c2, u8, 128)
    u8 = layers.Concatenate()([u8, c2_att])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    c1_att = dual_cross_attention(c1, u9, 64)
    u9 = layers.Concatenate()([u9, c1_att])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs, outputs)
    return model
