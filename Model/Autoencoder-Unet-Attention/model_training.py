from get_mammography_dataset import get_mammography_dataset
from model import unet_autoencoder_dca

MODEL_SAVE_PATH = "autoencoder_dca.keras"

def train_model():
    autoencoder_dca = unet_autoencoder_dca()
    autoencoder_dca.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    (x_ray_train, y_roi_train), (x_ray_test, y_roi_test) = get_mammography_dataset()

    autoencoder_dca.fit(x_ray_train, y_roi_train, epochs=35, batch_size=20, validation_data=(x_ray_test, y_roi_test))
    autoencoder_dca.save(MODEL_SAVE_PATH)
    return autoencoder_dca
