#1. Define the task
#2. Prepare the data
#3. Define the model
#5. Error analysis
#6. Train the model
#7. Preformance evaluation
#8. Validation and testing
#9. Corrective measures

import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Normalization, Dense, InputLayer # type: ignore
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber # type: ignore
from tensorflow.keras.metrics import RootMeanSquaredError # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore



##DATA PREPARATION
data = pd.read_csv(filepath_or_buffer="train.csv", delimiter=",")

#print out the 5 first rows of the data
#data.head()

#plot the data for all the possible 2d combinations

#sns.pairplot(data[['years', 'km', 'rating', 'condition', 'economy', 'top speed', 'hp', 'torque', 'current price']], diag_kind="kde")
#plt.show()

#transform the data into a tensor and casting it to reduce decimal places
tensor_data = tf.cast(tf.constant(data), tf.float32)


#shuffle the data to prevent it of be biased

tensor_data = tf.random.shuffle(tensor_data)




##DEFINE THE MODEL

X = tensor_data[:,3:-1]
y = tensor_data[:, -1]
#to ensure the rigth dimensions
y = tf.expand_dims(y, axis=-1) 

#Normalization layer x_normalized = (x - mean) / sqrt(variance) 
normalizer = Normalization()
#instead of determine the mean and variance for each column, the layer will calculate it
normalizer.adapt(X)
normalizer(X)


##LINEAR REGRESSION MODEL

#The sequential model is a linear stack of layers. It means that the output of the previous layer is the input of the next layer
#The Dense layer permites to connect all the neurons of the previous layer to the next layer


#model = tf.keras.sequential()
#model.add(normalizer)
#model.add(Dense(units=1))

#or 

model = tf.keras.Sequential([
    InputLayer(shape=(8,)),
    normalizer,
    Dense(units=1),
])

model.summary()

#The losse functions are used to measure the error of the model. The optimizer is used to minimize the loss function

#Using mean squared error as loss function can be problamatic when some points have a high error. 
#In this case, the model will try to minimize the error of the points with high error and the points with low error will be ignored. 
#To solve this problem, we can use the mean absolute error as loss function

#The huber loss function is a combination of the mean squared error and the mean absolute error. 
# It is less sensitive to outliers than the mean squared error and more sensitive than the mean absolute error

#The optimizer is used to minimize the loss function. The Adam optimizer is a popular optimizer that is used to minimize the loss 
# function. It works by updating the weights of the model in the direction that reduces the loss function
 
model.compile(
    optimizer= Adam(learning_rate=100.0), 
    loss=MeanAbsoluteError(),
    metrics=[RootMeanSquaredError()]
    )


##TRAIN THE MODEL AND OPTIMIZE 

#The fit method is used to train the model. The epochs parameter is used to determine the number of times the model will see the data.
#The verbose parameter is used to determine the amount of information that will be displayed during the training process

history = model.fit(X, y, epochs=100, verbose=1)

#print(history.history)

plt.plot(history.history['loss'])
plt.title('Model loss') 
plt.ylabel('Loss')
plt.xlabel('Epoch') 
plt.legend(['Train'])
plt.show()

##PERFORMANCE EVALUATION

#We can use the Root Mean Squared Error to evaluate the performance of the model. It works by taking the square root of the mean 
# squared error. We place it in the model.compile method to evaluate the performance of the model

plt.plot(history.history['root_mean_squared_error'])
plt.title('Model root mean squared error') 
plt.ylabel('Root mean squared error')
plt.xlabel('Epoch') 
plt.legend(['Train'])
plt.show()


##VALIDATION AND TESTING

#Instead of using the whole dataset to train the model, we can split the dataset into training, validation, and testing sets.

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
DATASET_SIZE = len(X)

X_train = X[:int(TRAIN_RATIO * DATASET_SIZE)]
y_train = y[:int(TRAIN_RATIO * DATASET_SIZE)]

X_val = X[int(TRAIN_RATIO * DATASET_SIZE):int((TRAIN_RATIO + VAL_RATIO) * DATASET_SIZE)]
y_val = y[int(TRAIN_RATIO * DATASET_SIZE):int((TRAIN_RATIO + VAL_RATIO) * DATASET_SIZE)]

X_test = X[int((TRAIN_RATIO + VAL_RATIO) * DATASET_SIZE):]
y_test = y[int((TRAIN_RATIO + VAL_RATIO) * DATASET_SIZE):]

normalizer = Normalization()
normalizer.adapt(X_train)

model = tf.keras.Sequential([
    InputLayer(shape=(8,)),
    normalizer,
    Dense(units=1),
])

model.compile(
    optimizer= Adam(learning_rate=100.0), 
    loss=MeanAbsoluteError(),
    metrics=[RootMeanSquaredError()]
    )

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, verbose=1)

y_true = list(y_test[:,0].numpy())

y_pred = list(model.predict(X_test)[:,0])



model.predict(X_test)

ind = np.arange(100)
plt.figure(figsize=(40, 12))

width = 0.4

plt.bar(ind, y_true, width, label='Predicted Car Price')
plt.bar(ind + width, y_pred, width, label='True Car Price')

plt.xlabel('Actual vs Predicted Car Price')
plt.ylabel('Car Price')

plt.legend(['True Car Price', 'Predicted Car Price'])

plt.show()

##CORRECTIVE MEASURES

#we can use hidden layers to improve the performance of the model. 
#The hidden layers are used to learn the complex patterns in the data.
#By adding more Dense layers, we can increase the complexity of the model and improve its performance.
#The activation function is used to introduce non-linearity in the model.
#There is a wide range of activation functions like the sigmoid, tanh, relu, and leaky relu functions.
#-Sigmoide: 1/(1+e^-x)
#-Tanh: (e^x - e^-x)/(e^x + e^-x)
#-Relu: max(0,x)
#-Leaky Relu: max(0.01x, x)

model_with_hidden_layers = tf.keras.Sequential([
    InputLayer(shape=(8,)),
    normalizer,
    Dense(units=32, activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=1),
])

model_with_hidden_layers.compile(
    optimizer= Adam(learning_rate=.1), 
    loss=MeanAbsoluteError(),
    metrics=[RootMeanSquaredError()]
    )

history = model_with_hidden_layers.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, verbose=1)

plt.plot(history.history['loss'])
plt.title('Model root mean squared error') 
plt.ylabel('Root mean squared error')
plt.xlabel('Epoch') 
plt.legend(['Train'])
plt.show()

y_true = list(y_test[:,0].numpy())

y_pred = list(model.predict(X_test)[:,0])



model.predict(X_test)

ind = np.arange(100)
plt.figure(figsize=(40, 12))

width = 0.4

plt.bar(ind, y_true, width, label='Predicted Car Price')
plt.bar(ind + width, y_pred, width, label='True Car Price')

plt.xlabel('Actual vs Predicted Car Price')
plt.ylabel('Car Price')

plt.legend(['True Car Price', 'Predicted Car Price'])

plt.show()