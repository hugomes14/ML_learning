import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TKAgg")
import tensorflow_datasets as tfds 
import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall, AUC #type: ignore
from tensorflow.keras.layers import InputLayer, Conv2D, Dense, MaxPool2D, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D, Resizing, Rescaling, RandomFlip, RandomRotation #type: ignore
from tensorflow.keras.losses import BinaryCrossentropy #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from tensorflow.keras.callbacks import Callback, EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau #type: ignore
from tensorflow.keras.regularizers import L2, L1 #type: ignore
import sklearn
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns


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
    image = tf.image.resize(image, (IM_SIZE, IM_SIZE)) / 255.0
    return image, label

resize_rescale_layers = tf.keras.Sequential([
    Resizing(IM_SIZE, IM_SIZE),
    Rescaling(1.0/255)
])



def augment(image, label):
    #augmented_image = tf.image.rot90(image)
    #augmented_image = tf.image.adjust_saturation(augmented_image, saturation_factor=0.3)
    augmented_image = tf.image.flip_left_right(image)
    
    return resize_rescale(augmented_image, label)


augment_layers = tf.keras.Sequential([
    RandomRotation(factor= (0.25, 0.2501)),
    RandomFlip(mode= 'HORIZONTAL',)
])



def augment_layer(image, label):
    return augment_layers(resize_rescale_layers(image), training= True), label

train_dataset = (train_dataset
                 .shuffle(buffer_size = 8, reshuffle_each_iteration=True)
                 .map(augment_layer)
                 .batch(32)
                 .prefetch(tf.data.AUTOTUNE))

val_dataset = (val_dataset
               .shuffle(buffer_size = 8, reshuffle_each_iteration=True)
               .map(resize_rescale)
               .batch(32)
               .prefetch(tf.data.AUTOTUNE))

test_dataset = (test_dataset
                .map(resize_rescale)
                .batch(1))


lenet_model = tf.keras.Sequential([

    InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),


    Conv2D(filters= 6, kernel_size=3, strides=1, padding='same', activation='relu')#, kernel_regularizer=L2(0.01)),
    BatchNormalization(),
    MaxPool2D(pool_size=2, strides=2),
    #Dropout(rate=0.3),


    Conv2D(filters= 16, kernel_size=3, strides=1, padding='same', activation='relu')#, kernel_regularizer=L2(0.01)),
    BatchNormalization(),
    MaxPool2D(pool_size=2, strides=2),
    #Dropout(rate=0.3),


    Flatten()


    Dense(100, activation='relu')#, kernel_regularizer=L2(0.01)),
    BatchNormalization(),
    #Dropout(rate=0.4),
    
    Dense(50, activation='relu')#, kernel_regularizer=L2(0.01)),
    BatchNormalization(),


    Dense(1, activation='sigmoid')#, kernel_regularizer=L2(0.01))
])
print(lenet_model.summary())

#-----------------Callbacks-------------------------- 

class LossCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        print("/n Epock Number {} the model has a loss of {}".format(epoch+1, logs["loss"]))
    
    def on_batch_end(self, batch, logs):
        print("/n Epock Number {} the model has a loss of {}".format(batch, logs) )


csv_callback = CSVLogger(
    'logs.csv', separator= ',', append= True
)

es_callback = EarlyStopping(
    monitor= 'val_loss', min_delta= 0, patience= 3,
    verbose= 0, mode= 'auto', baseline= None, restore_best_weights= False
)

def scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return float(lr*tf.math.exp(-0.1))
    
scheduler_callback = LearningRateScheduler(scheduler, verbose= 0.9)

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
    optimizer = Adam(learning_rate = 0.1),
    loss = BinaryCrossentropy(),
    metrics = [BinaryAccuracy(name= 'accuracy'), Precision(name= 'precision'), Recall(name= 'recall'), AUC(name= 'acu')]
)

history = lenet_model.fit(train_dataset, validation_data= val_dataset, epochs = 10, verbose = 1, callbacks= [plateau_callback, checkpoint_callback])

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

print(lenet_model.evaluate(test_dataset, verbose = 1))

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


#-------------Save the model----------------

#we can save the model using the save method. The model is saved in the .h5 format. The model can be loaded using the load_model method.

#lenet_model.save('malaria_model.h5')

