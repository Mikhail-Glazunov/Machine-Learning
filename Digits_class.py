#Importing Packages
import numpy as np
from matplotlib import pyplot as plt
#Importing keras with tensorflow backend
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras import losses
from keras.layers import Activation
from keras import optimizers
from keras.layers import LeakyReLU
#Creating swish activation function
from keras import backend as k
from keras.utils.generic_utils import get_custom_objects
'__name__'
class Swish(Activation):
    def __init__(self,activation, **kwargs):
        super(Swish,self).__init__(activation,**kwargs)
        self.__name__='Swish'
def swish(x):
    return (k.sigmoid(x)*x)
get_custom_objects().update({'Swish': Swish(swish)})
#Loading MNIST data
(X_train, y_train),(X_test, y_test)=mnist.load_data()
#Reshaping MNIST data
X_train=(X_train.reshape(60000,784)).astype('float32')
X_test=(X_test.reshape(10000,784)).astype('float32')
#Normalizing data
X_test=X_test/255
X_train=X_train/255
#Setting up output layer
n_outputs=10
Y_train=np_utils.to_categorical(y_train,n_outputs)
Y_test=np_utils.to_categorical(y_test,n_outputs)
#Building hidden layers
layer=Sequential()
#Hidden layer 1
layer.add(Dense(300,input_shape=(784,)))
layer.add(Activation('relu'))
layer.add(Dropout(0.2))
#Hidden layer 2
layer.add(Dense(300,input_shape=(784,)))
layer.add(Activation('relu'))
layer.add(Dropout(0.4))
#Building output layer
layer.add(Dense(10))
layer.add(Activation('softmax'))
#Compiling the model
layer.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
#Training the model
results=layer.fit(X_train,Y_train,batch_size=256,epochs=10,verbose=2,validation_data=(X_test,Y_test))
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
plt.legend(['train', 'test'], loc='upper right')
plt.grid()
plt.show()
#Printing results
metrics=layer.evaluate(X_test, Y_test, verbose=2)
predict=layer.predict_classes(X_test)
correct=np.nonzero(predict==y_test)[0]
incorrect=np.nonzero(predict!=y_test)[0]
print()
print("        MODEL RESULTS\n"
    "*******************************\n"
    "* Test data accuracy:         *\n"
    "*",round(metrics[1],3),"                      *"
    "\n* Test data loss:             *\n"
    "*",round(metrics[0],3),"                      *"
    "\n* Correct classifcation:      *\n"
    "*",len(correct),"                       *"
    "\n* Incorrect classification:   *\n"
    "*",len(incorrect),"                        *"
    "\n*******************************")
