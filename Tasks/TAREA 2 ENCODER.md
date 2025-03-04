### Investigar bloque de:

##### ResNet (Residual Block)

Un **bloque ResNet (Residual Block)** es la unidad básica de las redes neuronales residuales ( **ResNet** ), una arquitectura propuesta por **Microsoft Research** en 2015 en el paper  *"Deep Residual Learning for Image Recognition"* .

¿Por qué es importante?

- Evita el problema del desvanecimiento del gradiente en redes profundas.
- Permite entrenar modelos muy profundos con cientos de capas.
- Usa conexiones residuales (skip connections) para mejorar la propagación de la información.

**Implementación en keras:**

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Input
from tensorflow.keras.models import Model

def residual_block(inputs, filters, kernel_size=3, stride=1):
    """
    Implementa un bloque residual básico en ResNet.

    Parámetros:
    - inputs: Entrada de la capa anterior.
    - filters: Número de filtros en las capas convolucionales.
    - kernel_size: Tamaño del kernel (por defecto 3x3).
    - stride: Tamaño del paso en la convolución.

    Retorna:
    - La salida del bloque residual.
    """
  
    # Primera capa convolucional
    x = Conv2D(filters, kernel_size, strides=stride, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
  
    # Segunda capa convolucional
    x = Conv2D(filters, kernel_size, strides=1, padding="same")(x)
    x = BatchNormalization()(x)
  
    # Skip Connection (Conexión Residual)
    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding="same")(inputs)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = inputs
  
    # Suma de la conexión residual
    x = Add()([x, shortcut])
    x = ReLU()(x)
  
    return x

```

**Explicación de la función:**

- Se aplican dos capas convolucionales con normalización por lotes (BatchNormalization) y función de activación ReLU.
- La conexión residual (shortcut) salta el bloque convolucional y se suma al resultado.
- Si las dimensiones de entrada y salida no coinciden, se usa una convolución 1x1 para ajustar el tamaño.

**Implementación de red con bloques residuales en keras:**

```python
def build_resnet(input_shape=(32, 32, 3), num_classes=10):
    inputs = Input(shape=input_shape)
  
    # Primera capa convolucional
    x = Conv2D(64, (3,3), padding="same", activation="relu")(inputs)
  
    # Agregar dos bloques residuales
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
  
    # Capa final de clasificación
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
  
    # Definir el modelo
    model = Model(inputs, x)
  
    return model

# Crear el modelo
model = build_resnet()

# Mostrar la arquitectura
model.summary()
```

**Explicación:**

- Se inicia con una convolución básica.
- Se agregan dos bloques residuales con 64 filtros.
- Se usa GlobalAveragePooling2D() para reducir la dimensionalidad antes de la salida.
- La última capa es una densa (softmax) para clasificación en 10 clases.


### Investigar arquitectura de:

##### VGG 16

VGG16 es una arquitectura de red neuronal convolucional (CNN) desarrollada por el Visual Geometry Group (VGG) de la Universidad de Oxford. Fue presentada en el artículo "Very Deep Convolutional Networks for Large-Scale Image Recognition" y obtuvo buenos resultados en el desafío ImageNet en 2014. VGG16 se caracteriza por su profundidad y simplicidad en la estructura, utilizando principalmente capas convolucionales con **filtros de tamaño 3 × 3**, **max pooling de 2×2** y capas completamente conectadas al final. Tiene 16 capas con pesos entrenables (**13 convolucionales y 3 completamente conectadas**), lo que da origen a su nombre. Se usa comúnmente en tareas de visión por computadora, como clasificación de imágenes, detección de objetos y segmentación, y es popular en transfer learning debido a su efectividad en la extracción de características.

**Implementación de VGG16 en keras:**

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Definir la arquitectura manualmente
def build_vgg16():
    model = Sequential([
        # Bloque 1
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        # Bloque 2
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        # Bloque 3
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        # Bloque 4
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        # Bloque 5
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        # Capas completamente conectadas
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(1000, activation='softmax')  # 1000 clases de ImageNet
    ])
  
    return model

# Construimos el modelo
model = build_vgg16()

# Cargar los pesos preentrenados de ImageNet
weights_path = tf.keras.utils.get_file(
    "vgg16_weights_tf_dim_ordering_tf_kernels.h5",
    "https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
)
model.load_weights(weights_path)

# Cargar una imagen de prueba
img_path = tf.keras.utils.get_file("elephant.jpg", "https://upload.wikimedia.org/wikipedia/commons/6/6a/Indian_Elephant.jpg")

# Preprocesar la imagen
img = image.load_img(img_path, target_size=(224, 224))  
img_array = image.img_to_array(img)  
img_array = np.expand_dims(img_array, axis=0)  
img_array = preprocess_input(img_array)  

# Mostrar la imagen
plt.imshow(img)
plt.axis("off")
plt.show()

# Realizar la predicción
preds = model.predict(img_array)

# Decodificar y mostrar los resultados
decoded_preds = decode_predictions(preds, top=3)[0]
for i, (imagenet_id, label, score) in enumerate(decoded_preds):
    print(f"{i+1}. {label}: {score:.4f}")

```

##### Google Net Inception

Google Inception (también conocido como GoogLeNet) es una familia de arquitecturas de redes neuronales convolucionales (CNN) desarrolladas por Google para tareas de visión por computadora, especialmente clasificación de imágenes. La primera versión, Inception v1, se presentó en 2014 en el artículo "Going Deeper with Convolutions".

Características de Inception:

- Bloques Inception: Permiten el uso de convoluciones de diferentes tamaños en paralelo, mejorando la eficiencia.
- Reducción del costo computacional: Emplea capas de convolución 1x1 para reducir la dimensionalidad antes de aplicar convoluciones más grandes.
- Uso de módulos auxiliares: Se incluyen cabezales auxiliares en capas intermedias para mejorar el aprendizaje.

A lo largo de los años, la arquitectura ha evolucionado con versiones como Inception v2, v3, v4 y Inception-ResNet.

**Implementación de inception v1 en keras:**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# Definición de un bloque Inception
def inception_block(x, filters):
    f1, f3r, f3, f5r, f5, proj = filters
  
    # Ramas del Inception
    branch1 = layers.Conv2D(f1, (1,1), activation='relu', padding='same')(x)

    branch2 = layers.Conv2D(f3r, (1,1), activation='relu', padding='same')(x)
    branch2 = layers.Conv2D(f3, (3,3), activation='relu', padding='same')(branch2)

    branch3 = layers.Conv2D(f5r, (1,1), activation='relu', padding='same')(x)
    branch3 = layers.Conv2D(f5, (5,5), activation='relu', padding='same')(branch3)

    branch4 = layers.MaxPooling2D((3,3), strides=1, padding='same')(x)
    branch4 = layers.Conv2D(proj, (1,1), activation='relu', padding='same')(branch4)

    # Concatenación de las ramas
    output = layers.concatenate([branch1, branch2, branch3, branch4], axis=-1)
    return output

# Construcción de la Red Inception
def build_inception_v1(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(64, (7,7), strides=2, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((3,3), strides=2, padding='same')(x)
  
    x = layers.Conv2D(64, (1,1), activation='relu')(x)
    x = layers.Conv2D(192, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((3,3), strides=2, padding='same')(x)

    # Bloques Inception
    x = inception_block(x, [64, 96, 128, 16, 32, 32])
    x = inception_block(x, [128, 128, 192, 32, 96, 64])
    x = layers.MaxPooling2D((3,3), strides=2, padding='same')(x)

    x = inception_block(x, [192, 96, 208, 16, 48, 64])
    x = inception_block(x, [160, 112, 224, 24, 64, 64])
    x = inception_block(x, [128, 128, 256, 24, 64, 64])
    x = inception_block(x, [112, 144, 288, 32, 64, 64])
    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = layers.MaxPooling2D((3,3), strides=2, padding='same')(x)

    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = inception_block(x, [384, 192, 384, 48, 128, 128])

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Definir el modelo
input_shape = (224, 224, 3)  # Tamaño de entrada estándar
num_classes = 1000  # Número de clases para ImageNet

model = build_inception_v1(input_shape, num_classes)
model.summary()

```

**Inception - Resnet**

Inception-ResNet es una variante de la arquitectura Inception que combina bloques Inception con conexiones residuales (de ResNet). Fue introducida en el artículo "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning" en 2016.

Principales Características:

- Combinación de Inception y ResNet:
- Usa bloques Inception para extraer características en múltiples escalas.
- Usa conexiones residuales para facilitar la propagación del gradiente, acelerando el entrenamiento y evitando el problema del desvanecimiento del gradiente.

Dos versiones:

- Inception-ResNet-v1: Menos profundo y más rápido.
- Inception-ResNet-v2: Más profundo, mejor precisión pero más costoso computacionalmente.

**Implementación Inception-Resnet en keras:**

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

# Definición del Bloque Residual de Inception
def inception_resnet_block(x, filters, scale=0.1):
    f1, f3r, f3, f5r, f5 = filters

    branch1 = layers.Conv2D(f1, (1,1), activation='relu', padding='same')(x)

    branch2 = layers.Conv2D(f3r, (1,1), activation='relu', padding='same')(x)
    branch2 = layers.Conv2D(f3, (3,3), activation='relu', padding='same')(branch2)

    branch3 = layers.Conv2D(f5r, (1,1), activation='relu', padding='same')(x)
    branch3 = layers.Conv2D(f5, (3,3), activation='relu', padding='same')(branch3)
    branch3 = layers.Conv2D(f5, (3,3), activation='relu', padding='same')(branch3)

    # Concatenar todas las ramas
    merged = layers.concatenate([branch1, branch2, branch3], axis=-1)

    # Proyección con convolución 1x1
    shortcut = layers.Conv2D(tf.keras.backend.int_shape(x)[-1], (1,1), padding='same')(merged)

    # Residual Connection: salida = entrada + shortcut * escala
    x = layers.Add()([x, shortcut * scale])
    x = layers.Activation('relu')(x)
    return x

# Construcción de la Red Inception-ResNet
def build_inception_resnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Bloques iniciales
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((3,3), strides=2, padding='same')(x)

    # Bloques Inception-ResNet
    x = inception_resnet_block(x, [32, 32, 32, 32, 32])
    x = inception_resnet_block(x, [64, 64, 64, 64, 64])
    x = layers.MaxPooling2D((3,3), strides=2, padding='same')(x)

    x = inception_resnet_block(x, [128, 128, 128, 128, 128])
    x = inception_resnet_block(x, [256, 256, 256, 256, 256])
    x = layers.MaxPooling2D((3,3), strides=2, padding='same')(x)

    x = inception_resnet_block(x, [512, 512, 512, 512, 512])
    x = inception_resnet_block(x, [1024, 1024, 1024, 1024, 1024])

    # Clasificación
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Crear el modelo
input_shape = (224, 224, 3)  # Tamaño estándar de imagen
num_classes = 1000  # Número de clases (ej. ImageNet)

model = build_inception_resnet(input_shape, num_classes)
model.summary()

```


##### Dense Net

DenseNet (Densely Connected Convolutional Networks) es una arquitectura de redes neuronales convolucionales (CNN) introducida por Huang et al. en 2017. Su principal característica es que cada capa está conectada directamente con todas las capas anteriores en una estructura de "conexión densa". A diferencia de otras arquitecturas como ResNet, donde se usan conexiones residuales para saltar ciertas capas, en DenseNet cada capa recibe como entrada las características de todas las capas anteriores y pasa su salida a todas las capas posteriores.

Esto tiene varias ventajas:

- Mejor propagación de la información y del gradiente.
- Menos parámetros que otras arquitecturas porque reutiliza las características.
- Mayor eficiencia en el entrenamiento.

**Implementación DenseNet en keras:**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dense_block(x, num_layers, growth_rate):
    """Crea un bloque denso con varias capas convolucionales."""
    for _ in range(num_layers):
        y = layers.BatchNormalization()(x)
        y = layers.ReLU()(y)
        y = layers.Conv2D(4 * growth_rate, (1, 1), padding="same", use_bias=False)(y)  # Bottleneck layer
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        y = layers.Conv2D(growth_rate, (3, 3), padding="same", use_bias=False)(y)  # 3x3 Conv layer
    
        x = layers.Concatenate()([x, y])  # Conexión densa
    
    return x

def transition_layer(x, compression=0.5):
    """Reduce las dimensiones del feature map para mejorar eficiencia."""
    num_filters = int(x.shape[-1] * compression)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(num_filters, (1, 1), padding="same", use_bias=False)(x)
    x = layers.AveragePooling2D((2, 2), strides=2, padding="same")(x)
    return x

def build_densenet(input_shape=(32, 32, 3), num_classes=10, growth_rate=12, num_blocks=3, num_layers_per_block=4):
    """Construye una versión simplificada de DenseNet."""
    inputs = keras.Input(shape=input_shape)
  
    # Capa inicial
    x = layers.Conv2D(2 * growth_rate, (3, 3), padding="same", use_bias=False)(inputs)
  
    # Bloques densos con capas de transición
    for _ in range(num_blocks - 1):
        x = dense_block(x, num_layers_per_block, growth_rate)
        x = transition_layer(x)
  
    # Último bloque sin capa de transición
    x = dense_block(x, num_layers_per_block, growth_rate)
  
    # Capa final
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
  
    model = keras.Model(inputs, outputs, name="DenseNet")
    return model

# Crear modelo
model = build_densenet()
model.summary()
```

Explicación del Código

- dense_block(x, num_layers, growth_rate)
  - Contiene varias capas convolucionales de crecimiento (growth_rate).
  - Cada nueva capa concatena su salida con todas las anteriores.
- transition_layer(x, compression=0.5)
  - Reduce el número de filtros mediante una convolución 1x1.
  - Se aplica un AveragePooling2D para reducir las dimensiones espaciales.
- build_densenet(...)
  - Inicia con una convolución 3x3.
  - Alterna bloques densos y capas de transición.
  - Termina con una capa GlobalAveragePooling2D y una densa para clasificación.


### Estudiar:

##### Capa convolucional:

Recorre la matriz que representa la imagen mediante un filtro de tamaño nxn y un stride
para extraer un numero determinado de imágenes caracteristicas.

Ejemplo:

```python
conv_layer = Conv2D(
	filters=32,          # Número de filtros
	kernel_size=(3, 3),  # Tamaño del filtro (kernel)
	strides=(1, 1),      # Desplazamiento del filtro
	padding='same',      # Relleno (valid/same)
	activation='relu',   # Función de activación
	input_shape=(64, 64, 3)  # Tamaño de la imagen de entrada (alto, ancho, canales)
)
```

##### Capa Max pooling:

se usa para reducir el tamaño espacial (altura y ancho) de los mapas de características
extraídos por las capas convolucionales, reteniendo la información más importante y
reduciendo la cantidad de cálculos.

Ejemplo:

```python
maxpool_layer = MaxPooling2D(
	pool_size=(2, 2),  # Tamaño de la ventana de pooling
	strides=(2, 2),    # Paso del pooling (stride)
	padding='valid'    # Padding ('valid' o 'same')
)
```

##### Capa de normalización

###### Batch Normalization (BatchNormalization)

Normaliza los valores de activación en cada mini-lote.

Esta capa normaliza las activaciones por mini-lote, restando la media y dividiendo por la desviación estándar, lo que acelera el entrenamiento y estabiliza la red.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

batch_norm_layer = BatchNormalization(
	axis=-1,   # Normaliza sobre los canales (última dimensión)
	momentum=0.99,  # Controla qué tan rápido se actualizan la media y varianza
	epsilon=0.001   # Pequeño valor para evitar división por cero
)

model = Sequential([
	Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
	BatchNormalization(),  # Normaliza la salida de la convolución
	Flatten(),
	Dense(64, activation='relu'),
	BatchNormalization(),  # También se puede usar en capas densas
	Dense(10, activation='softmax')
])

model.summary()
```

**Explicación de los Parámetros:**

| Parametro     | Descipción                                                                                            |
| ------------- | ------------------------------------------------------------------------------------------------------ |
| axis=-1       | Normaliza a lo largo de los canales (útil en imágenes con formato (batch, height, width, channels)). |
| momentum=0.99 | Controla la actualización de la media y varianza acumuladas.                                          |
| epsilon=0.001 | Pequeño valor para estabilidad numérica.                                                             |

###### Layer Normalization (LayerNormalization)

Normaliza los valores en cada muestra individualmente. Esta capa normaliza cada muestra individualmente, a diferencia de BatchNormalization, que lo hace por mini-lotes.

```python
from tensorflow.keras.layers import LayerNormalization

layer_norm_layer = LayerNormalization(
	axis=-1,    # Normaliza cada muestra a lo largo de la última dimensión
	epsilon=1e-5  # Pequeño valor para evitar división por cero
)

model = Sequential([
	Dense(128, activation='relu', input_shape=(20,)),
	LayerNormalization(),
	Dense(64, activation='relu'),
	LayerNormalization(),
	Dense(10, activation='softmax')
])

model.summary()
```

**Cuándo usar LayerNormalization?**

Cuando el tamaño del mini-lote es muy pequeño o variable.
En Redes Transformers y NLP, donde la normalización por capa es más estable.

###### Normalization (Normalization)

Normaliza los datos de entrada basándose en la media y desviación estándar globales. Esta capa es útil para preprocesar los datos de entrada, escalándolos en función de la media y desviación estándar calculadas antes del entrenamiento.

Ejemplo:

```python
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import numpy as np

normalization_layer = Normalization()

# Crear una capa de normalización
norm_layer = Normalization()

# Datos de entrada de ejemplo (10 muestras, 5 características)
data = np.random.rand(10, 5)

# Ajustar la normalización con los datos de entrenamiento
norm_layer.adapt(data)

# Aplicar la normalización a nuevos datos
normalized_data = norm_layer(data)
print(normalized_data.numpy())
```

Import:

Uso Típico:
Se ajusta la capa con datos de entrenamiento (adapt()).
Se aplica la normalización a nuevos datos.

Capa de atención.(Opcional)

Capa de atención multifractal.(Opcional)

### Conceptos:

##### Dataset

Un dataset es un conjunto de ejemplos que se usa para entrenar, validar y evaluar un modelo de machine learning o deep learning.

Tipos de datasets según su uso:

Dataset de Entrenamiento → Se usa para ajustar los pesos de la red neuronal.
Dataset de Validación → Evalúa el rendimiento del modelo durante el entrenamiento para ajustar hiperparámetros.
Dataset de Prueba → Se usa después del entrenamiento para medir el rendimiento final del modelo.

Dataset	Tipo de Datos	Uso
MNIST	Imágenes de dígitos (28x28)	Clasificación de dígitos
CIFAR-10	Imágenes de objetos (32x32)	Clasificación de objetos
COCO	Imágenes con etiquetas	Detección de objetos
Imagenet	Imágenes de alta resolución	Clasificación de imágenes
PASCAL VOC	Imágenes con bounding boxes	Segmentación y detección


##### Arquitectura de la red

La arquitectura de una red neuronal define cómo están organizadas y conectadas las capas dentro del modelo.

Componentes Claves de la Arquitectura:

Capas → Elementos básicos de la red (Convolucionales, Densas, Pooling, etc.).
Conexiones → Cómo fluye la información entre las capas.
Activaciones → Funciones que introducen no linealidad (ReLU, Sigmoid, etc.).
Parámetros → Pesos y sesgos que la red ajusta en el entrenamiento.

Arquitectura	Características	Aplicaciones
MLP (Perceptrón Multicapa):	Capas densas conectadas	Clasificación simple
LeNet-5:					Primera red CNN (5 capas) Reconocimiento de dígitos
AlexNet:					CNN profunda con ReLU Clasificación de imágenes
VGG16/VGG19:			CNN con capas convolucionales profundas	Visión por computadora
ResNet:					Usa conexiones residuales (skip connections)	Modelos muy profundos
Transformer:				Atención en paralelo, sin convoluciones	NLP y visión computacional


##### Capas

Las capas son los bloques fundamentales de las redes neuronales. Cada capa realiza operaciones específicas sobre los datos de entrada.
