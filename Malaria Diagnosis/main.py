#Malaria is a life-threatening disease caused by parasites that are transmitted to people through the bites of infected
# In 2019, there were an estimated 229 million cases of malaria worldwide.
# The estimated number of malaria deaths stood at 409 000 in 2019.
# Children aged under 5 years are the most vulnerable group affected by malaria; in 2019, they accounted for 67% (274 000) 
# of all malaria deaths worldwide.
# The WHO African Region carries a disproportionately high share of the global malaria burden. In 2019,
# the region was home to 94% of malaria cases and deaths.

#We start loading the data
#Then visualize the data
#Preprocess the data
#Build the model
#Train the model
#Evaluate the model

#1. task
#2. Data loading
#3. Model building
#4. Error analysis
#5. Training and optimization
#6. Performance evaluation
#7. Validating and testing
#8. Corrective measures

#-------------Task understanding----------------


#We will build a model based on convulotion neural network that can predict if a cell image has malaria or not.
#The input data is a cell image and the output is a binary classification of 0 or 1. 0 means the cell image does not have malaria 
# and 1 means the cell image has malaria.
#when mosquitoe bites a person, the parasite is transmitted to the person's blood. 
#The parasite then travels to the liver where it matures and reproduces.
# the blood samples can be prepeared by using a thin smear of blood or a thick smear of blood.
#We will use the thin smear of blood to train the model.

#Using the image of the thin smear of blood, we can segment the image to get the cell image. And then evaluate the cell image to 
# determine if it has malaria or not.

#-------------Data Preparation----------------

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds 
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv2D, Dense, MaxPool2D, Flatten, BatchNormalization #type: ignore
from tensorflow.keras.losses import BinaryCrossentropy #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore


dataset, dataset_info = tfds.load('malaria', with_info=True, as_supervised=True, shuffle_files=True, split=['train']) 

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def splits(dataset, train_ratio, val_ratio, test_ratio):
    total_size = dataset_info.splits['train'].num_examples
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size)
    return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = splits(dataset[0], TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

#print(list(train_dataset.take(1).as_numpy_iterator()), list(val_dataset.take(1).as_numpy_iterator()), list(test_dataset.take(1).as_numpy_iterator()))


#-------------Data Visualization----------------
"""
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_dataset.take(16)):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(image.numpy())
    plt.axis("off")
    plt.title("Infected" if label.numpy() == 1 else "Uninfected")
plt.tight_layout()
plt.show()
"""
#-------------Data Preprocessing----------------

#.1 Resizing

#The images in the dataset have different sizes. We need to resize the images to a fixed size so that we can feed them into the model.

IM_SIZE = 224

def resize_rescale(image, label):
    return tf.image.resize(image, (IM_SIZE, IM_SIZE))/255.0, label

train_dataset = train_dataset.map(resize_rescale)
val_dataset = val_dataset.map(resize_rescale)
test_dataset = test_dataset.map(resize_rescale)

"""
for image, label in train_dataset.take(1):
    print(image, label)
"""

#Now, suffle, batc and prefetch the dataset
#Is important to ensure that train_data, val_data, and test_data dimensions are compatible

train_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.batch(1)

#2. Normalization/stadardization

#In the cars dataset, the values of each collumn are close to each other and follow a normal distribution. In this case, 
# we can use the standardization technique to normalize the data. X = (X - mean) / std

#however in images dataset, the pixel values may not follow a normal distribution. 
# In this case, we can use the min-max scaling technique to normalize the data. X = (X - min) / (max - min)


#-------------Model ConvNet Building----------------

#the input shape of the image is (224, 224, 3). Which means we have 150528 features. Using Dense layers the number of parameters tend 
# to be very large because every neuron in the current layer is connected to every neuron in the previous layer.
#To reduce the number of parameters, we can use the convolutional neural network (CNN) model. In CNN, we use the convolutional layer,
# that unlike the dense layer, the neurons in the convolutional layer are connected to only a few neurons in the previous layer.
#The convolutional layer use the filter to extract the features from the image. The filter or a kernel is a matrix that slides over 
# the image. The filter is multiplied with the image pixel values and the result is stored in the feature map. The dimension of the 
#kernel enables to extract different features from the image. Large kernel size is used to extract the global features from the image
# and small kernel size is used to extract the local features from the image. The output size of the feature map is determined by the
# formula: (n - f + 2p) / s + 1, where n is the input size, f is the filter size, p is the padding size, and s is the stride size. 
#The type of kernell enhnaces some features and reduces others. The outline kernell enhances the edges of the image. The sharpen
#The blur kernell reduces the noise in the image. The emboss kernell enhances the edges of the image.
# The filter is then moved to the next position and the process is repeated until the filter has covered the entire image. The feature 
# map is then passed to the activation function. The activation function is used to introduce non-linearity to the 
# model. The activation function is applied to the feature map to get the output of the convolutional 
# layer. 
# 
# The output of the convolutional layer is then passed to the pooling layer.
# The pooling layer is used to reduce the dimensionality of the feature map. The pooling layer takes the maximum value
# from the feature map and stores it in the pooled feature map. The pooling layer is used to reduce the number of parameters in the 
# model and to reduce the computation time. Max pooling is used to reduce the dimensionality of the feature map. The max pooling
# takes the maximum value from the feature map and stores it in the pooled feature map. The average pooling takes the average value 
# from the feature map and stores it in the pooled feature map. The global average pooling takes the average value from the feature
# map and stores it in the pooled feature map. The pooled feature map is then passed to the next convolutional layer. The process is
# repeated until the feature map is passed to the dense layer. The dense layer is used to classify the image. The dense layer takes
# the feature map and flattens it to a vector. 

#Using TelNet model

#Conv2D: filters - the number of filters in the convolutional layer
#        kernel_size - the size of the filter
#        strides - the number of pixels the filter moves in each step
#        padding - the padding used to keep the input size the same as the output size 
#        activation - the activation function used to introduce non-linearity to the model

#MaxPool2D: pool_size - the size of the pooling layer
#           strides - the number of pixels the pooling layer moves in each step


lenet_model = tf.keras.Sequential([
        InputLayer(input_shape = (IM_SIZE, IM_SIZE, 3)),
        Conv2D(filters = 6, kernel_size = 3, strides = 1, padding = 'valid', activation = 'relu'),
        BatchNormalization(),
        MaxPool2D(pool_size = 2, strides = 2),
        Conv2D(filters = 16, kernel_size = 3, strides = 1, padding = 'valid', activation = 'relu'),
        BatchNormalization(),
        MaxPool2D(pool_size = 2, strides = 2),
        Flatten(),
        Dense(100, activation = 'relu'),
        BatchNormalization(),
        Dense(10, activation = 'relu'),
        BatchNormalization(),
        Dense(1, activation = 'sigmoid')
    ])

print(lenet_model.summary())

#-------------Binary Crossentropy Loss----------------

#Move to error sanctioanlization. The binary crossentropy loss is used to calculate the loss of the model. Its formula is
# -y * log(y_hat) - (1 - y) * log(1 - y_hat), where y is the true label and y_hat is the predicted label. The binary crossentropy

#the model accuracy is calculted by the formula: (TP + TN) / (TP + TN + FP + FN), where TP is the true positive,
# TN is the true negative, FP is the false positive, and FN is the false negative. 

lenet_model.compile(
    optimizer = Adam(learning_rate = 0.001),
    loss = BinaryCrossentropy(),
    metrics = ['accuracy']
)

history = lenet_model.fit(train_dataset, validation_data= val_dataset, epochs = 10, verbose = 1)
"""
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy') 
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
"""

#-------------Model Evaluation----------------

print(lenet_model.evaluate(test_dataset))


"""
lenet_model.predict(test_dataset.take(1))

def parasite_or_not(x):
    if x > 0.5:
        return 1
    else:
        return 0
    
for i, (image, label) in enumerate(test_dataset.take(9)):
    ax = plt.subplot(3,3, i+1)
    plt.imshow(image[0])
    plt.axis("off")
    plt.title(str(parasite_or_not(label.numpy()[0])) + ":" + str(parasite_or_not(lenet_model.predict(image)[0])))
    plt.tight_layout()
    plt.axis("off")
    plt.show()
"""

#-------------Save the model----------------

#we can save the model using the save method. The model is saved in the .h5 format. The model can be loaded using the load_model method.

lenet_model.save('malaria_model.h5')

