import tensorflow as tf
from tensorflow.keras import layers, Model
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split

NUM_CLASSES = 47
IMG_SHAPE = (28, 28, 1)
BATCH_SIZE = 128

# -----------------------
# Model
# -----------------------
def build_emnist_classifier():
    inputs = layers.Input(shape=IMG_SHAPE)

    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return Model(inputs, outputs)


model = build_emnist_classifier()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------
# Load EMNIST
# -----------------------
def load_emnist_balanced(mat_path):
    mat = scipy.io.loadmat(mat_path)
    data = mat['dataset']

    train = data['train'][0,0]
    images = train['images'][0,0]
    labels = train['labels'][0,0]

    images = images.reshape((-1, 28, 28), order='F')
    images = (images.astype(np.float32) / 127.5) - 1.0
    images = np.expand_dims(images, axis=-1)

    labels = labels.astype(np.int32).squeeze()

    return images, labels


# -----------------------
# Load data
# -----------------------
images, labels = load_emnist_balanced("matlab/emnist-balanced.mat")

# -----------------------
# Train / Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# -----------------------
# tf.data pipelines
# -----------------------
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -----------------------
# Train
# -----------------------
model.fit(
    train_ds,
    epochs=20,
    validation_data=test_ds
)

# -----------------------
# Save model
# -----------------------
model.save("models/emnist_classifier.keras")