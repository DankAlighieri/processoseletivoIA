import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Input

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

(xTrain, yTrain), (xTest, yTest) = keras.datasets.mnist.load_data()

# dividindo em treino, validação e teste
xTrain, xVal, yTrain, yVal = train_test_split(xTrain, yTrain, stratify=yTrain, test_size = 0.25, random_state=42)

# Arrays de 4 dimensões (batch, altura, largura, canais)
xTrain = xTrain.reshape(xTrain.shape[0], 28, 28, 1)
xVal = xVal.reshape(xVal.shape[0], 28, 28, 1)
xTest = xTest.reshape(xTest.shape[0], 28, 28, 1)

# interface de entraadas
inputShape = (28,28,1)

# Normalizando os dados do modelo (0-1)
xTrain = xTrain.astype('float32') / 255.0
xVal = xVal.astype('float32') / 255.0
xTest = xTest.astype('float32') / 255.0

# Construindo a CNN

model = Sequential()

# primeiro bloco conv2d -> batchnorm -> relu -> maxpooling
model.add(Input(shape=inputShape))
model.add(Conv2D(
    32, # filtros
    (3, 3), # kernel
    padding='same',
    use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

# segundo bloco conv2d -> batchnorm -> relu -> maxpooling
model.add(Conv2D(
    64,
    (3, 3), 
    padding='same',
    use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

# terceiro bloco conv2d -> batchnorm -> relu -> maxpooling
model.add(Conv2D(
    128, 
    (3, 3),
    padding='same',
    use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

# blocos de saida
model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax'))

model.summary()