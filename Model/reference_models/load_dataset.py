import pickle
import random
import numpy as np

def getMammographyDataset():
    # Cargar el dataset en formato .pkl
    file_path = '~/lcruizDev/Data/dataset_320_512_CBIS-DDSM_dict.pkl'
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)

    # Obtener y mezclar las claves del diccionario aleatoriamente
    keys = list(dataset.keys())
    random.shuffle(keys)  # Aleatorizar los datos

    # Definir el índice de división (70% entrenamiento, 30% prueba)
    split_index = int(len(keys) * 0.7)

    # Dividir claves en entrenamiento y prueba
    train_keys = keys[:split_index]
    test_keys = keys[split_index:]

    # Extraer datos de entrenamiento y prueba
    x_ray_train = np.array([dataset[key]['image_file_numpy'] for key in train_keys])
    y_roi_train = np.array([dataset[key]['roi_mask_file_numpy'] for key in train_keys])

    x_ray_test = np.array([dataset[key]['image_file_numpy'] for key in test_keys])
    y_roi_test = np.array([dataset[key]['roi_mask_file_numpy'] for key in test_keys])

    # Asegurar que los datos tengan la forma adecuada (num_samples, height, width, channels)
    if len(x_ray_train.shape) == 3:  # Si las imágenes son en escala de grises sin canal
        x_ray_train = np.expand_dims(x_ray_train, axis=-1)
        x_ray_test = np.expand_dims(x_ray_test, axis=-1)

        y_roi_train = np.expand_dims(y_roi_train, axis=-1)
        y_roi_test = np.expand_dims(y_roi_test, axis=-1)

    x_ray_train =  x_ray_train / 255
    x_ray_test = x_ray_test / 255

    y_roi_train = y_roi_train / 255
    y_roi_test = y_roi_test / 255

    return (x_ray_train, y_roi_train), (x_ray_test, y_roi_test)
