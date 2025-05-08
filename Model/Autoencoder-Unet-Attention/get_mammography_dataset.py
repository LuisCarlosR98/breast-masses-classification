import pickle
import random
import numpy as np

def get_mammography_dataset():
    file_path = '..\dataset\dataset_256_256_CBIS-DDSM_dict.pkl'
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)

    keys = list(dataset.keys())
    random.shuffle(keys)

    split_index = int(len(keys) * 0.7)

    train_keys = keys[:split_index]
    test_keys = keys[split_index:]

    x_ray_train = np.array([dataset[key]['image_file_numpy'] for key in train_keys])
    y_roi_train = np.array([dataset[key]['roi_mask_file_numpy'] for key in train_keys])

    x_ray_test = np.array([dataset[key]['image_file_numpy'] for key in test_keys])
    y_roi_test = np.array([dataset[key]['roi_mask_file_numpy'] for key in test_keys])

    return (x_ray_train, y_roi_train), (x_ray_test, y_roi_test)
