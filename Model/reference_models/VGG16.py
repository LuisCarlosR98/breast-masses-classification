import pickle
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from load_dataset import getMammographyDataset

MODEL_SAVE_PATH = "../../lcruizDev/Models/vgg16_autoencoder.h5"

#Constantes para el tamaño de las imágenes
HEIGHT = 512
WIDTH = 320

def build_vgg16_encoder(input_shape = (HEIGHT, WIDTH, 1)):
    input_img = layers.Input(shape = input_shape)
    
    # Bloque 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=2, padding='same')(x)
    
    # Bloque 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=2, padding='same')(x)
    
    # Bloque 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=2, padding='same')(x)
    
    # Bloque 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=2, padding='same')(x)
    
    # Bloque 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), strides=2, padding='same')(x)
    
    return models.Model(input_img, encoded, name="vgg16_encoder")

def build_vgg16_decoder(input_shape = (16, 10, 512)):
    encoded_input = layers.Input(shape=(16, 10, 512))   # Ajusta las dimensiones según la salida del encoder
    x = layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='same')(encoded_input)
    x = layers.UpSampling2D((2, 2))(x)  
    x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)  
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)  
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)  
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)

    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x) 
    
    return models.Model(encoded_input, decoded, name="decoder")

# Obtener el dataset}
(x_ray_train, y_roi_train), (x_ray_test, y_roi_test) = getMammographyDataset()
print(x_ray_train.shape, y_roi_train.shape)

# Construcción del autoencoder
input_img = layers.Input(shape=(HEIGHT, WIDTH, 1))
encoder = build_vgg16_encoder(input_shape=(HEIGHT, WIDTH, 1))
encoded_output = encoder(input_img)

# Construcción del decodificador
decoder = build_vgg16_decoder(input_shape=(16, 10, 512))
decoded_output = decoder(encoded_output)

#Construcción autoencoder
autoencoder = models.Model(input_img, decoded_output, name="vgg16_autoencoder")

# Resumen del modelo
autoencoder.summary()

# Compilar el modelo
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Entrenamiento del autoencoder
autoencoder.fit(x_ray_train, y_roi_train, epochs=10, batch_size=40, validation_data=(x_ray_test, y_roi_test))

#Guardar el modelo entrenado
autoencoder.save(MODEL_SAVE_PATH)

