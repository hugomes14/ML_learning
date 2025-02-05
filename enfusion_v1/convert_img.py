import tensorflow as tf

# Path to the image dataset
image_dir = "dataset"

# Parameters
image_size = (128, 128)  # Resize images to 128x128
batch_size = 32         # Batch size

# Load and preprocess the dataset
raw_dataset = tf.keras.utils.image_dataset_from_directory(
    image_dir,
    image_size=image_size,
    batch_size=None,  # Do not batch yet, process as individual samples
    shuffle=True  # Shuffle the data
)

# Normalize pixel values to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

# Apply normalization and ensure the format is preserved
tf_dataset = raw_dataset.map(lambda x, y: (normalization_layer(x), y))

# Shuffle the data
shuffled_dataset = tf_dataset.shuffle(buffer_size=8)

# Convert to tuples (image, label) for custom batching if required
final_dataset = shuffled_dataset.map(lambda x, y: (x, y))


# Save the dataset (optional)
tf.data.Dataset.save(final_dataset, "saved_tf_dataset")


