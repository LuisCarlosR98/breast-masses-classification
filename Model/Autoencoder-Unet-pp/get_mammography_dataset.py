import pickle
import random
import numpy as np

def getMammographyDataset(file_path):

    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)

    keys = list(dataset.keys())
    random.shuffle(keys)

    x_ray = np.array([dataset[key]['image_file_numpy'] for key in keys])
    y_roi = np.array([dataset[key]['roi_mask_file_numpy'] for key in keys])

    y_roi_train_bin = (y_roi > 0.5).astype(np.uint8)

    y_roi = y_roi_train_bin.astype(np.float32)

    return (x_ray, y_roi)
