import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv2D, Dense, MaxPool2D, Flatten, BatchNormalization, GlobalAveragePooling2D #type: ignore
from tensorflow.keras.losses import BinaryCrossentropy #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from tensorflow.keras.callbacks import Callback, EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau #type: ignore
from tensorflow.keras.metrics import BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall, AUC #type: ignore
from tensorflow.keras.regularizers import L2, L1 #type: ignore
import sklearn
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import numpy as np

dataset = tf.data.Dataset.load("saved_tf_dataset")
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def splits(dataset, train_ratio, val_ratio, test_ratio):
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size)
    return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = splits(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

print(train_dataset, len(val_dataset), len(test_dataset))


IM_SIZE = 16

def resize_rescale(image, label):
    image = tf.image.resize(image, (IM_SIZE, IM_SIZE))
    return image, label

def augment(image, label):
    augmented_image = tf.image.random_flip_up_down(image)
    
    return resize_rescale(augmented_image, label)

def augment2(image, label):
    augmented_image = tf.image.random_brightness(image, 0.2)

    return resize_rescale(augmented_image, label)

def augment3(image, label):
    augmented_image = tf.image.random_flip_left_right(image)

    return resize_rescale(augmented_image, label)


train_dataset = (train_dataset
                 .map(resize_rescale)
                 .concatenate(train_dataset.map(augment))
                 .concatenate(train_dataset.map(augment2))
                 .concatenate(train_dataset.map(augment3))
                 #.map(augment)
                 .shuffle(buffer_size = 8, reshuffle_each_iteration=True)
                 .batch(32)
                 .prefetch(tf.data.AUTOTUNE))

val_dataset = (val_dataset
               .map(resize_rescale)
               .shuffle(buffer_size = 8, reshuffle_each_iteration=True)
               .batch(32)
               .prefetch(tf.data.AUTOTUNE))

test_dataset = (test_dataset
                .map(resize_rescale)
                .batch(1))




#-------------Model ConvNet Building----------------


lenet_model = tf.keras.Sequential([
        InputLayer(input_shape = (IM_SIZE, IM_SIZE, 3)),
        Conv2D(filters = 8, kernel_size = 3, strides = 1, padding = 'valid', activation = 'relu'),
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


checkpoint_callback = ModelCheckpoint(
    'checkpoints.keras', monitor= 'val_loss', verbose= 1, save_best_only= True,
    save_weights_only= False, mode= 'auto', save_freq= 'epoch'
)

plateau_callback = ReduceLROnPlateau(
    monitor= 'val_accuracy', fator= 0.5, patience= 2, verbose= 1
)

#-------------Binary Crossentropy Loss----------------
metrics = [TruePositives(name= 'tp'), FalsePositives(name= 'fp'), TrueNegatives(name= 'tn'), FalseNegatives(name= 'fn'), 
           BinaryAccuracy(name= 'accuracy'), Precision(name= 'precision'), Recall(name= 'recall'), AUC(name= 'acu')]

lenet_model.compile(
    optimizer = Adam(learning_rate = 1.0),
    loss = BinaryCrossentropy(),
    metrics = [BinaryAccuracy(name= 'accuracy'), Precision(name= 'precision'), Recall(name= 'recall'), AUC(name= 'acu')]
)

history = lenet_model.fit(train_dataset, validation_data= val_dataset, epochs = 40, verbose = 1, callbacks= [plateau_callback, checkpoint_callback])

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


#-------------Model Evaluation----------------

print(lenet_model.evaluate(test_dataset))

labels = []
inp = []

for x,y in test_dataset.as_numpy_iterator():
    labels.append(y)
    inp.append(x)

#Convert the labels in a vector 
labels = np.array([i[0] for i in labels])

#Convert the predicted in a vector similiar to labels
predicted = lenet_model.predict(np.array(inp)[:,0,...])[:,0]


threshold = 0.5

cm = confusion_matrix(labels, predicted > threshold)
print(cm)


plt.figure(figsize=(8, 8))

sns.heatmap(cm, annot= True)
plt.title('Confusion matrix - {}'.format(threshold))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()



fp, tp,thresholds = roc_curve(labels, predicted)
plt.plot(fp, tp)
plt.xlabel("False positive rate")
plt.ylabel("True Positive rate")
plt.grid()
for i in range(0, len(thresholds), 20):
    plt.text(fp[i], tp[i], thresholds[i])

plt.show()

# Save the entire model to a file
model_path = "infusion_model_v1.h5"
lenet_model.save(model_path)
print(f"Model saved to {model_path}")

