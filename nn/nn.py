import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
# Hyperparameters
NB_EPOCHS = 15
BATCH_SIZE = 32
NB_CLASSES = 10
VALIDATION_SPLIT = 0.2

# Load Data
# X_train is 60000 rows of 28*28 values, X_test is 10000 rows of 28*28
(X_train, y_train),(X_test, y_test) = mnist.load_data()

# flatten
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize
X_train /= 255
X_test /= 255

# Convert class into one-hot encoding
Y_train = keras.utils.to_categorical(y_train, NB_CLASSES)
Y_test = keras.utils.to_categorical(y_test, NB_CLASSES)

# Build the model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
# model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=SGD(), metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = NB_EPOCHS, verbose = 1,
                   validation_data=(X_test,Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print(score)