import numpy as np
import matplotlib.pyplot as plt 
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.layers import InputLayer, Input, Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, Layer #type: ignore
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

#-------------Model Creation----------------

#We can create costum layers by subclassing the Layer class and implementing the __init__ and __call__ methods.
class NeurallearnDense(Layer):
    def __init__(self, output_units, activation = None):
        super(NeurallearnDense, self).__init__()
        self.output_units = output_units
        self.activation = activation

    def build(self, input_shape):
        self.w = self.add_weight(shape = (input_shape[-1], self.units), initializer = 'random_normal', trainable = True)
        self.b = self.add_weight(shape = (self.units,), initializer = 'zeros', trainable = True)
    
    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w) + self.b)



functional_lenet_model = "Model(inputs = func_input, outputs = output,  name = 'Lenet_Model')"
 
print(functional_lenet_model.summary()) 


#-------------Model Training----------------


functional_lenet_model.compile(
    optimizer = Adam(learning_rate = 0.001),
    loss = BinaryCrossentropy(),
    metrics = ['accuracy']
)

history = functional_lenet_model.fit(train_dataset, validation_data= val_dataset, epochs = 1, verbose = 1)


#-------------Model Evaluation----------------

print(functional_lenet_model.evaluate(test_dataset))
