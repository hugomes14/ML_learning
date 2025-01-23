import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
import numpy as np
import cv2
import time
import statistics

# Load and recompile model
model_path = "infusion_model_v1.h5"
lenet_model = tf.keras.models.load_model(model_path)
lenet_model.compile(optimizer=Adam(learning_rate=0.0001), 
                    loss=BinaryCrossentropy(), 
                    metrics=['accuracy'])

print("Model loaded and recompiled successfully!")


video = cv2.VideoCapture("teste2.avi")
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)


window_size = 12  
x_stride = 12
y_stride = 12  
IM_SIZE = 32 

def preprocess_window(window):
    """Resize and normalize a window."""
    return tf.image.resize(window, (IM_SIZE, IM_SIZE))/ 255.0

def sliding_window_fast(image, window_size, x_stride, y_stride):
    """Generate sliding windows and their locations using NumPy."""
    h, w, c = image.shape  

    
    num_windows_y = (h - window_size) // y_stride + 1
    num_windows_x = (w - window_size) // x_stride + 1

    
    windows = np.lib.stride_tricks.as_strided(
        image,
        shape=(num_windows_y, num_windows_x, window_size, window_size, c),
        strides=(
            y_stride * image.strides[0],  # Stride along rows
            x_stride * image.strides[1],  # Stride along columns
            image.strides[0],           # Stride within a window (row step)
            image.strides[1],           # Stride within a window (column step)
            image.strides[2],           # Channel step
        ),
        writeable=False
    )

    
    windows = windows.reshape(-1, window_size, window_size, c)

    
    y_coords, x_coords = np.meshgrid(
        np.arange(0, h - window_size + 1, y_stride),
        np.arange(0, w - window_size + 1, x_stride),
        indexing='ij'
    )
    locations = np.stack((x_coords.ravel(), y_coords.ravel()), axis=-1)

    return windows, locations


previous_time = time.time()
begging=[]
sliding_window=[]
preprocess=[]
predi = []
squares = []
show_im=[]
mean_fps = []

while True:
    # Start of frame processing
    frame_start_time = time.time()

    # Read the next frame
    ret, frame = video.read()
    if not ret:
        break  # Exit if no more frames

    image = frame

    # Record the beginning step time
    step_start = time.time()
    windows, locations = sliding_window_fast(image, window_size, x_stride, y_stride)
    sliding_window_time = time.time() - step_start

    # Preprocess windows
    step_start = time.time()
    windows = preprocess_window(windows)
    preprocess_time = time.time() - step_start

    # Perform batch prediction
    step_start = time.time()
    predictions = lenet_model.predict(windows, verbose=0)
    prediction_time = time.time() - step_start

    # Draw rectangles for predictions > 0.5
    step_start = time.time()
    for i, prediction in enumerate(predictions):
        if prediction > 0.5:
            x, y = locations[i]
            cv2.rectangle(frame, (x, y), (x + window_size, y + window_size), (0, 255, 0), 1)
    drawing_time = time.time() - step_start

    # Show the frame
    total_frame_time = time.time() - frame_start_time
    real_fps = 1 / total_frame_time
    cv2.putText(frame, f"FPS: {real_fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    step_start = time.time()
    cv2.imshow("Detection", frame)
    display_time = time.time() - step_start

    # Calculate total frame processing time and FPS
    total_frame_time = time.time() - frame_start_time
    real_fps = 1 / total_frame_time

    # Display FPS on the frame
    

    # Store timings for averaging later
    begging.append(total_frame_time)
    sliding_window.append(sliding_window_time)
    preprocess.append(preprocess_time)
    predi.append(prediction_time)
    squares.append(drawing_time)
    show_im.append(display_time)
    mean_fps.append(real_fps)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"Frame Processing Time: {statistics.mean(begging):.5f}s")
        print(f"Sliding Window Time: {statistics.mean(sliding_window):.5f}s")
        print(f"Preprocessing Time: {statistics.mean(preprocess):.5f}s")
        print(f"Prediction Time: {statistics.mean(predi):.5f}s")
        print(f"Drawing Time: {statistics.mean(squares):.5f}s")
        print(f"Display Time: {statistics.mean(show_im):.5f}s")
        print(f"Mean FPS: {statistics.mean(mean_fps):.2f}")
        break

# Cleanup
video.release()
cv2.destroyAllWindows()
