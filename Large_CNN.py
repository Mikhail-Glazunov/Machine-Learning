#Importing packages
from matplotlib import pyplot as plt
#Importing keras with tensorflow backend
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
#Importing convolutional keras packages
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
#Loading MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#Reshaping MNIST data to 28x28
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
#Normalizing MNIST data
X_train = X_train / 255
X_test = X_test / 255
#Configuring output layer
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
#Building the model
layer = Sequential()
#Convolutional layer 1
layer.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
#Pooling layer 1
layer.add(MaxPooling2D())
#Convolutional layer 2
layer.add(Conv2D(15, (3, 3), activation='relu'))
#Pooling layer 2
layer.add(MaxPooling2D())
#Dropout layer with 20% probability
layer.add(Dropout(0.2))
#Flattaning layer
layer.add(Flatten())
#Connected layer 1 with 128 nodes
layer.add(Dense(128, activation='relu'))
#Connect layer2 with 50 nodes
layer.add(Dense(50, activation='relu'))
#Output layer
layer.add(Dense(num_classes, activation='softmax'))
#Compiling the model
layer.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Training the model
results=layer.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200,verbose=2)
#Plotting model accuracy
plt.plot(results.history['accuracy'],"r")
plt.plot(results.history['val_accuracy'],"k")
plt.title('Accuracy of Model')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.grid()
plt.show()
#Plotting model loss
plt.plot(results.history['loss'],"r")
plt.plot(results.history['val_loss'],"k")
plt.title('Loss of Model')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.grid()
plt.show()
#Printing results
metrics=layer.evaluate(X_test, y_test, verbose=2)
print()
print("        MODEL RESULTS\n"
    "*******************************\n"
    "* Test data accuracy:         *\n"
    "*",round(metrics[1],4),"                      *"
    "\n* Test data loss:             *\n"
    "*",round(metrics[0],4),"                     *"
    "\n*******************************")

