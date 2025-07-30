## CARGA DEL MODELO
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from get_mammography_dataset import get_mammography_dataset

def loadModel(model_path = "autoencoder_dca.keras"):
    """
    Carga un modelo Keras desde un archivo.

    Parámetros:
    - model_path: str, ruta al archivo del modelo.

    Retorna:
    - model: keras.Model, el modelo cargado.
    """
    return keras.models.load_model(model_path)


def generatePredictions(model, x_test):
    """
    Genera predicciones utilizando un modelo Keras.

    Parámetros:
    - model: keras.Model, el modelo a utilizar para la predicción.
    - x_test: np.ndarray, datos de entrada para la predicción.

    Retorna:
    - predictions: np.ndarray, las predicciones generadas por el modelo.
    """
    decoded_imgs_dca = model.predict(x_test)
    predictions_bin_dca = (decoded_imgs_dca > 0.5).astype(np.uint8)
    return predictions_bin_dca

def showImages(original, generated, roi, n=15):
    """
    Muestra imágenes originales, generadas y ROI.

    Parámetros:
    - original: np.ndarray, imágenes originales.
    - generated: np.ndarray, imágenes generadas.
    - roi: np.ndarray, imágenes de ROI.
    - n: int, número de imágenes a mostrar.
    """
    HEIGHT = 256
    WIDTH = 256

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Muestra la imagen original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(original[i].reshape(HEIGHT, WIDTH), cmap="gray")
        plt.title("Original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Muestra la imagen reconstruida
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(generated[i].reshape(HEIGHT, WIDTH), cmap="gray")
        plt.title("Generada")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Muestra la imagen original del roi
        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(roi[i].reshape(HEIGHT, WIDTH), cmap="gray")
        plt.title("Original Roi")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

def generateDiceCoefficient(y_true, y_pred):
    """
    Calcula el coeficiente de Dice entre dos máscaras binarias.

    Parámetros:
    - y_true: np.ndarray, máscara de referencia (ground truth).
    - y_pred: np.ndarray, máscara predicha.

    Retorna:
    - coeficiente de Dice (float).
    """
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)

    if y_true.shape != y_pred.shape:
        raise ValueError("Las máscaras deben tener la misma forma.")

    intersection = np.logical_and(y_true, y_pred).sum()
    total = y_true.sum() + y_pred.sum()

    if total == 0:
      print('vacias')
      return 1.0  # Ambas máscaras están vacías

    return 2.0 * intersection / total

def calculateDiceScores(x_ray_test, y_roi_test, predictions_bin_dca):
    """
    Calcula los puntajes de Dice para un conjunto de imágenes.

    Parámetros:
    - x_ray_test: np.ndarray, imágenes de prueba.
    - y_roi_test: np.ndarray, máscaras de referencia (ground truth).
    - predictions_bin_dca: np.ndarray, máscaras predichas.

    Retorna:
    - max_dice_score: float, el puntaje de Dice máximo.
    """
    dice_scores_dca = []

    for i in range(len(x_ray_test)):
        y_true = y_roi_test[i].squeeze()
        y_pred = predictions_bin_dca[i].squeeze()

        dice = generateDiceCoefficient(y_true, y_pred)

        dice_scores_dca.append(dice)

    return dice_scores_dca
