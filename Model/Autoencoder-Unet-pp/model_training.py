from get_mammography_dataset import getMammographyDataset
from model import unet_plus_plus
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle

MODEL_SAVE_PATH = "autoencoder_pp.keras"
HISTORY_SAVE_PATH = "history_autoencoder_pp.pkl"
FILE_PATH_TRAINING = '..\dataset\dataset_256_256_CBIS-DDSM_dict-training.pkl'
FILE_PATH_TEST = '..\dataset\dataset_256_256_CBIS-DDSM_dict-test.pkl'

def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def train_model():
    # Obtener el dataset TRAINING
    (x_ray_train, y_roi_train) = getMammographyDataset(FILE_PATH_TRAINING)

    # Divide entrenamiento en 80% train y 20% validación
    x_ray_train, x_ray_val, y_roi_train, y_roi_val = train_test_split(x_ray_train, y_roi_train, test_size=0.2, random_state=42)

    autoencoder_pp = unet_plus_plus()
    autoencoder_pp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    metrics=[dice_coef, iou_coef]
    autoencoder_pp.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

    history = autoencoder_pp.fit(x_ray_train, y_roi_train, epochs=40, batch_size=20, validation_data=(x_ray_val, y_roi_val))

    # GUARDAR MODELO
    autoencoder_pp.save(MODEL_SAVE_PATH)
    print(f"✅ Modelo guardado en: {MODEL_SAVE_PATH}")

    # GUARDAR HISTORIAL DE MÉTRICAS
    with open(HISTORY_SAVE_PATH, "wb") as f:
        pickle.dump(history.history, f)
    print(f"✅ Historial guardado en: {HISTORY_SAVE_PATH}")
    return autoencoder_pp
