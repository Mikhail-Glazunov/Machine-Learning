import numpy as np
from matplotlib import pyplot as plt
import os
import time
import playsound
import speech_recognition as sr
from gtts import gTTS

#Dataset for training the network
dataset=[[3,   2.5, 1],
        [4,   1,   1],
        [4,   1.5, 1],
        [2.5,   .5,   0],
        [3.5, .5,  1],
        [2,   .5,  0],
        [5.5,  1.5,  1],
        [1,    1,  0],
        [2, 1.5, 0],
        [1.5, .5, 0],
        [5, 3, 1],
        [.5,1,0]]
#Randomly generated test value for prediction
p=np.random.uniform(0,5,1)
v=np.random.uniform(0,3,1)
test_value = [p,v]
#Scatter plot of dataset with test value
for i in range(len(dataset)):
    plt.grid()
    points=dataset[i][2]
    if points == 0:
        color = "b"
    else:
        color = "r"
    plt.scatter([dataset[i][0]], [dataset[i][1]], c=color)#training values
    plt.scatter([test_value[0]], [test_value[1]], c='black')#test value
plt.show()
#Defining an activation function (sigmoid)
def sigmoid(x):
    return 1/(1+np.exp(-x))
#Derivation of the sigmoid function
def dsigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))
#Training the nework
def training_nn():
    #Initialising weights and biases with normally distributed random numbers
    w1=np.random.randn()
    w2=np.random.randn()
    b=np.random.randn()
    #number of iterations
    iteration=10000
    #learning rate
    rate=0.1
    #measure progress of the network using the cost function
    costs = []
    #learnig loop
    for i in range(iteration):
        #To find a random point
        rand_int=np.random.randint(len(dataset))
        point = dataset[rand_int]
        z = w1*point[0] + w2*point[1] + b
        #Network prediction
        prediction  = sigmoid(z)
        #Target prediction
        target = point[2]
        #Cost Function
        cost_func=np.square(prediction-target)
        #Derivatve of cost function
        dcost_func=2*(prediction-target)
        dprediction=dsigmoid(z)
        #Derivation of weight and biases
        dw1=point[0]
        dw2=point[1]
        db= 1
        d_cost= dcost_func * dprediction
        dcdw1= d_cost*dw1
        dcdw2= d_cost*dw2
        dcdb= d_cost*db
        #Updating the weights and biases
        w1= w1-dcdw1*rate
        w2= w2-dcdw2*rate
        b= b-dcdb*rate
        #Plotting the progress of the cost
        if i % 100 == 0:
            c=0
            for j in range(len(dataset)):
                l = dataset[j]
                l_prediction = sigmoid(w1*l[0]+w2*l[1]+b)
                c+=np.square(l_prediction-l[2])
            costs.append(c)
    return costs, w1, w2, b
costs, w1, w2, b=training_nn()
#Plotting cost curve
plt.plot(costs,'r')
plt.show()
#Text to speech function
def speak(text):
    tts= gTTS(text=text,lang="en")
    name="prediction.mp3"
    tts.save(name)
    playsound.playsound(name)

#Test value prediction
z = w1*test_value[0]+w2*test_value[1]+b
prediction=sigmoid(z)
if prediction<0.5:
    speak("The point is blue")
    print("Prediction value: Blue",prediction)
else:
    speak("The point is red")
    print("Prediction value: Red",prediction)
        
