#At this moment, I have working with the sequential API. However, there are the functional API and the subclassing API. One limitation
#of the sequential API is that is not possible to, for example, besides to determinate if the cell is parasitized or not, 
#also determine its position in the image. This is possible with the functional API.

#The functional API is more flexible than the sequential API. It allows to create more complex models, 
#such as multi-output models, directed acyclic graphs, or models with shared layers.

#The subclassing API provides the most flexibility, but is also more complex and can be more difficult to debug. It is based 
# on object-oriented programming. Where the model is a class, and the layers are attributes of the class.

import numpy as np
import matplotlib.pyplot as plt 
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.layers import InputLayer, Input, Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization #type: ignore
from tensorflow.keras.losses import BinaryCrossentropy #type: ignore
from tensorflow.keras.metrics import Accuracy #type: ignore
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


#Now, suffle, batc and prefetch the dataset
#Is important to ensure that train_data, val_data, and test_data dimensions are compatible

train_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.batch(1)

#-------------Feature Extraction----------------

func_input = Input(shape = (IM_SIZE, IM_SIZE, 3), name = 'input_image')

x = Conv2D(filters = 6, kernel_size = 3, strides = 1, padding = 'valid', activation = 'relu')(func_input)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size = 2, strides = 2)(x)
x = Conv2D(filters = 16, kernel_size = 3, strides = 1, padding = 'valid', activation = 'relu')(x)
x = BatchNormalization()(x)
output = MaxPool2D(pool_size = 2, strides = 2)(x)

feature_extractor_model = Model(inputs = func_input, outputs = output,  name = 'Feature_Extractor')

#-----------------OR------------------------

feature_extractor_model = tf.keras.Sequential([
    Input(shape = (IM_SIZE, IM_SIZE, 3), name = 'input_image'),
    Conv2D(filters = 6, kernel_size = 3, strides = 1, padding = 'valid', activation = 'relu'),
    BatchNormalization(),
    MaxPool2D(pool_size = 2, strides = 2),
    Conv2D(filters = 16, kernel_size = 3, strides = 1, padding = 'valid', activation = 'relu'),
    BatchNormalization(),
    MaxPool2D(pool_size = 2, strides = 2)
])
 
print(feature_extractor_model.summary())


#-------------Model Creation----------------

x = feature_extractor_model(func_input)
x = Flatten()(x)
x = Dense(100, activation = 'relu')(x)
x = BatchNormalization()(x)
x = Dense(10, activation = 'relu')(x)
x = BatchNormalization()(x)
output = Dense(1, activation = 'sigmoid')(x)

functional_lenet_model = Model(inputs = func_input, outputs = output,  name = 'Lenet_Model')
 
print(functional_lenet_model.summary()) 


functional_lenet_model.compile(
    optimizer = Adam(learning_rate = 0.001),
    loss = BinaryCrossentropy(),
    metrics = ['accuracy']
)

history = functional_lenet_model.fit(train_dataset, validation_data= val_dataset, epochs = 1, verbose = 1)


#-------------Model Evaluation----------------

print(functional_lenet_model.evaluate(test_dataset))
