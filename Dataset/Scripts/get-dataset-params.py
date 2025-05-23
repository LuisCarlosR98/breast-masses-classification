import pandas as pd
import numpy as np
import pydicom
import pickle
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from typing import Optional

#Constantes para la normalización de las imágenes
HEIGHT = 256
WIDTH = 256
FILE_DATASET_NAME = f"dataset_{WIDTH}_{HEIGHT}_CBIS-DDSM_dict.pkl"
PATH_DATASET = f'../../../lcruizDev/Data/{FILE_DATASET_NAME}'

# Función para procesar el archivo CSV y filtrar las filas con 'image view' = 'MLO'
def process_csv(file_path: str) -> Optional[pd.DataFrame]:
    required_columns = {'patient_id', 'image view', 'image file path'}

    try:
        df = pd.read_csv(file_path)

        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Faltan las siguientes columnas requeridas: {', '.join(missing_columns)}")

        filtered_df = df[df['image view'].str.upper() == 'MLO']

        if filtered_df.empty:
            print("No se encontraron registros con 'image view' igual a 'MLO'.")

        return filtered_df

    except FileNotFoundError:
        print(f"❌ El archivo '{file_path}' no se encontró.")
    except pd.errors.EmptyDataError:
        print("❌ El archivo CSV está vacío.")
    except pd.errors.ParserError as e:
        print(f"❌ Error al parsear el archivo CSV: {e}")
    except ValueError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Ocurrió un error inesperado: {e}")
        
    return None

# 2. Funciones auxiliares
# Función para cargar un archivo DICOM y obtener la matriz de píxeles
def load_dicom(file_path: str) -> Optional[np.ndarray]:
    try:
        file_path = os.path.expanduser(file_path.strip())
        dicom = pydicom.dcmread(file_path)

        if dicom.get('SeriesDescription', '').lower() == 'cropped images':
            return None

        if not hasattr(dicom, 'pixel_array'):
            return None

        return dicom.pixel_array.astype(float)

    except FileNotFoundError:
        print(f"❌ Archivo no encontrado: {file_path}")
    except InvalidDicomError:
        print(f"❌ El archivo no es un DICOM válido: {file_path}")
    except Exception as e:
        print(f"❌ Error inesperado al cargar DICOM: {e}")

    return None

# Función para escalar una imagen
def gray_scale(image_array):
    scaled_image=(np.maximum(image_array,0)/image_array.max())*255
    scaled_image=np.uint8(scaled_image)
    scaled_image = applyCLAHE(scaled_image)
    return Image.fromarray(scaled_image)

def applyCLAHE(image_array):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(image_array)
    return clahe_image

# Función para normalizar una imagen
def normalize_image(image_array):
    image_array = image_array.astype(np.float32)

    min_val = np.min(image_array)
    max_val = np.max(image_array)
    range_val = max_val - min_val

    if range_val == 0:
        return np.zeros_like(image_array, dtype=np.float32)

    normalized_image = (image_array - min_val) / range_val
    return normalized_image

# Función para preprocesar una imagen
def preprocess_image(file_path):
    file_path_name = file_path.replace('000000.dcm', '1-1.dcm').replace('000001.dcm', '1-2.dcm')
    file_path_name = file_path_name.replace('\n', '')
    url = '~/Cancer/CBIS-DDSM/' + file_path_name
    image = load_dicom(url)
    if image is None:
        url = url.replace('1-2.dcm', '1-1.dcm')
        image = load_dicom(url)
    gray_image = gray_scale(image)
    resized_image = gray_image.resize((WIDTH, HEIGHT))
    np_rx = np.array(resized_image)
    np_rx = normalize_image(np_rx)
    np_rx = np.expand_dims(np_rx, axis = -1)
    return np_rx

# Función para mostrar una imagen
def printRx(img):
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

# Ejecución de las funciones
# Ruta al archivo CSV
csv_file_path = '~/Cancer/mass_case_description_train_set.csv'

# Llamar a la función y obtener el DataFrame filtrado
filtered_mlo_dataframe = process_csv(csv_file_path)
#filtered_mlo_dataframe.shape[0]
dic = {}
for index in tqdm(range(filtered_mlo_dataframe.shape[0])):
    patient_id = filtered_mlo_dataframe.iloc[index]['patient_id']
    orientation_breast = filtered_mlo_dataframe.iloc[index]['left or right breast']
    path_mm = filtered_mlo_dataframe.iloc[index]['image file path']
    path_roi_mask = filtered_mlo_dataframe.iloc[index]['ROI mask file path']
    file_procces = preprocess_image(path_mm)
    file_roi_mask_procces = preprocess_image(path_roi_mask)
    dic[patient_id + '_' + orientation_breast] = {
        'patient_id': patient_id,
        'left_or_right_breast': orientation_breast,
        'mass_shape': filtered_mlo_dataframe.iloc[index]['mass shape'],
        'pathology': filtered_mlo_dataframe.iloc[index]['pathology'],
        'image_file_numpy': file_procces,
        'roi_mask_file_numpy': file_roi_mask_procces
    }

# Crear los directorios si no existen
os.makedirs(os.path.dirname(PATH_DATASET), exist_ok=True)

with open(PATH_DATASET, "wb") as f:
    pickle.dump(dic, f)

print('Finishing.....')
