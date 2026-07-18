import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# insira seu código aqui

(xTrain, yTrain), (xTest, yTest) = keras.datasets.mnist.load_data()

# convertendo para float32 e normalizando
xTrain = xTrain.astype("float32") / 255.0
xTest = xTest.astype("float32") / 255.0

#adicionando grayscale (60000, 28, 28, 1)

xTrain = np.expand_dims(xTrain, axis=-1)
xTest = np.expand_dims(xTest, axis=-1)

rng = np.random.default_rng(42)
indices = rng.permutation(len(xTrain))

validationSize = int(len(xTrain) * 0.1)

validationIndices = indices[:validationSize]
trainIndices = indices[validationSize:]

xVal = xTrain[validationIndices]
yVal = yTrain[validationIndices]

xTrain = xTrain[trainIndices]
yTrain = yTrain[trainIndices]

print(xTrain.shape)
print(xVal.shape)
print(xTest.shape)