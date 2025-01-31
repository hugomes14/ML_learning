#There is other ways to evaluate the model performance other than accuracy, such as precision, recall, F1 score, and confusion matrix.
#Its is important to have an ideia about false positives and false negatives, and how they can affect the model performance.  
#The accuracy formula is defined as: Accuracy = (TP + TN) / (TP + TN + FP + FN); and tells us how many of the predictions are correct.
#The precision formula is defined as: Precision = TP / (TP + FP); and tells us how many of the positive predictions are correct.
#the recall formula is defined as: Recall = TP / (TP + FN); and tells us how many of the actual positive cases were predicted correctly.
#The F1 score formula is defined as: F1 = 2 * (Precision * Recall) / (Precision + Recall); and tells us the balance between precision and recall.
#The specificity formula is defined as: Specificity = TN / (TN + FP); and tells us how many of the actual negative cases were predicted correctly.
#The ROC curve is a graphical representation of the true positive rate (sensitivity) against the false positive rate (1-specificity). The objective is
#to have a curve here the TP rate is high and the FP rate is low. The AUC is the area under the curve and is a measure of how well the model is performing.
#When choosing the right threshold, we need to consider the trade-off between precision and recall. The ROC curve can help us to choose the right threshold.
#The AUC is a value between 0 and 1, where 1 means the model is perfect and 0.5 means the model is not better than random guessing. 
 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds 
import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall, AUC #type: ignore
from tensorflow.keras.layers import InputLayer, Conv2D, Dense, MaxPool2D, Flatten, BatchNormalization #type: ignore
from tensorflow.keras.losses import BinaryCrossentropy #type: ignore

from tensorflow.keras.optimizers import Adam #type: ignore


dataset, dataset_info = tfds.load('malaria', with_info=True, as_supervised=True, shuffle_files=True, split=['train']) 

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def splits(dataset, train_ratio, val_ratio, test_ratio):
    print(dataset)
    total_size = dataset_info.splits['train'].num_examples
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size)
    return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = splits(dataset[0], TRAIN_RATIO, VAL_RATIO, TEST_RATIO)


IM_SIZE = 224

def resize_rescale(image, label):
    return tf.image.resize(image, (IM_SIZE, IM_SIZE))/255.0, label

train_dataset = train_dataset.map(resize_rescale)
val_dataset = val_dataset.map(resize_rescale)
test_dataset = test_dataset.map(resize_rescale)



train_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.batch(1)


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
metrics = [TruePositives(name= 'tp'), FalsePositives(name= 'fp'), TrueNegatives(name= 'tn'), FalseNegatives(name= 'fn'), 
           BinaryAccuracy(name= 'accuracy'), Precision(name= 'precision'), Recall(name= 'recall'), AUC(name= 'acu')]

lenet_model.compile(
    optimizer = Adam(learning_rate = 0.001),
    loss = BinaryCrossentropy(),
    metrics = metrics
)

history = lenet_model.fit(train_dataset, validation_data= val_dataset, epochs = 5, verbose = 1)

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

#lenet_model.save('malaria_model.h5')

