import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time

# Load the saved model
model_path = "infuison_model_v1.h5"  # Or "saved_lenet_model"
lenet_model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")



# Recompile the model with the same configuration
lenet_model.compile(
    optimizer=Adam(learning_rate=0.001),  # Use the same optimizer
    loss=BinaryCrossentropy(),           # Use the same loss function
    metrics=['accuracy']                 # Specify metrics to track
)

print("Model recompiled successfully!")

video = cv2.VideoCapture("teste2.avi")
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fig, ax = plt.subplots()
img = ax.imshow(np.zeros((frame_height, frame_width)), cmap='gray', vmin=0, vmax=255)
plt.ion()  

previous_time = time.time()
while True:
    
    ret, frame = video.read()

    image = frame
    h, w, c = image.shape  # Get image dimensions

    window_size = 12  # Size of the sliding window
    stride = 12  # Step size for sliding window
    IM_SIZE = 224  # Resize to the required input size

    # Preprocessing function for the sliding window
    def preprocess_window(window):
        # Resize and normalize the window to the input size expected by the model
        window = tf.image.rgb_to_grayscale(image)
        window_resized = tf.image.resize(window, (IM_SIZE, IM_SIZE)) / 255.0
        return window_resized

    # Generate all sliding windows using tf.data.Dataset
    def sliding_window(image, window_size, stride, im_size):
        windows = []
        locations = []
        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                window = image[y:y + window_size, x:x + window_size, :]
                windows.append(window)
                locations.append([x, y])
        return windows, locations

    # Use the sliding_window function
    windows, locations = sliding_window(image, window_size, stride, IM_SIZE)

    # Convert to TensorFlow Dataset
    windows_dataset = tf.data.Dataset.from_tensor_slices(windows)

    # Map preprocessing function (resize and normalize)
    windows_dataset = windows_dataset.map(lambda window: preprocess_window(window), num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the windows and prefetch to improve performance
    batch_size = 32
    windows_dataset = windows_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Perform batch prediction
    predictions = lenet_model.predict(windows_dataset, verbose=0)

    # Draw rectangles for windows with prediction > 0.5
    for i, prediction in enumerate(predictions):
        if prediction > 0.5:
            x, y = locations[i]
            cv2.rectangle(image, (x, y), (x + window_size, y + window_size), (0, 255, 0), 1)

    # Display the resulting image
    now_time = time.time()
    real_fps = 1/(now_time-previous_time)
    cv2.putText(frame, f"FPS: {real_fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    img.set_data(image)
    plt.draw()
    previous_time = time.time()
    plt.pause(0.001)

