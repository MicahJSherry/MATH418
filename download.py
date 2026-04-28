import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Hyperparameters
# -----------------------
BATCH_SIZE = 128
LATENT_DIM = 100
NUM_CLASSES = 26
IMG_SHAPE = (28, 28, 1)
EPOCHS = 20

# -----------------------
# Load EMNIST Letters
# -----------------------
# EMNIST via keras doesn't exist natively → load from tfds
def load_emnist_balanced(mat_path):
    mat = scipy.io.loadmat(mat_path)
    data = mat['dataset']

    train = data['train'][0,0]
    images = train['images'][0,0]
    labels = train['labels'][0,0]

    images = images.reshape((-1, 28, 28), order='F')
    #images = np.transpose(images, (0, 2, 1))

    images = (images.astype(np.float32) / 127.5) - 1.0
    images = np.expand_dims(images, axis=-1)

    labels = labels.astype(np.int32).squeeze()

    return images, labels

# -----------------------
# Dataset
# -----------------------
images, labels = load_emnist_balanced("matlab/emnist-balanced.mat")

dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# -----------------------
# Generator
# -----------------------
def build_generator():
    noise = layers.Input(shape=(LATENT_DIM,))
    label = layers.Input(shape=(1,), dtype='int32')

    label_embedding = layers.Embedding(NUM_CLASSES, NUM_CLASSES)(label)
    label_embedding = layers.Flatten()(label_embedding)

    x = layers.Concatenate()([noise, label_embedding])

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(np.prod(IMG_SHAPE), activation='tanh')(x)

    img = layers.Reshape(IMG_SHAPE)(x)

    return tf.keras.Model([noise, label], img, name="Generator")

# -----------------------
# Discriminator
# -----------------------
def build_discriminator():
    img = layers.Input(shape=IMG_SHAPE)
    label = layers.Input(shape=(1,), dtype='int32')

    label_embedding = layers.Embedding(NUM_CLASSES, NUM_CLASSES)(label)
    label_embedding = layers.Flatten()(label_embedding)

    img_flat = layers.Flatten()(img)
    x = layers.Concatenate()([img_flat, label_embedding])

    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model([img, label], x, name="Discriminator")

# -----------------------
# Build models
# -----------------------
generator = build_generator()
discriminator = build_discriminator()

loss_fn = tf.keras.losses.BinaryCrossentropy()
g_optimizer = tf.keras.optimizers.Adam(0.0002)
d_optimizer = tf.keras.optimizers.Adam(0.0002)

# -----------------------
# Training step
# -----------------------
@tf.function
def train_step(images, labels):
    batch_size = tf.shape(images)[0]

    real = tf.ones((batch_size, 1))
    fake = tf.zeros((batch_size, 1))

    noise = tf.random.normal((batch_size, LATENT_DIM))
    random_labels = tf.random.uniform((batch_size, 1), 0, NUM_CLASSES, dtype=tf.int32)

    # -----------------
    # Train Generator
    # -----------------
    with tf.GradientTape() as tape:
        generated_images = generator([noise, random_labels], training=True)
        predictions = discriminator([generated_images, random_labels], training=True)
        g_loss = loss_fn(real, predictions)

    grads = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    # -----------------
    # Train Discriminator
    # -----------------
    with tf.GradientTape() as tape:
        real_preds = discriminator([images, tf.expand_dims(labels, -1)], training=True)
        fake_preds = discriminator([generated_images, random_labels], training=True)

        d_loss_real = loss_fn(real, real_preds)
        d_loss_fake = loss_fn(fake, fake_preds)
        d_loss = (d_loss_real + d_loss_fake) / 2

    grads = tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

    return g_loss, d_loss

# -----------------------
# Training loop
# -----------------------
def generate_samples(epoch):
    noise = tf.random.normal((26, LATENT_DIM))
    labels = tf.range(0, 26, dtype=tf.int32)
    labels = tf.expand_dims(labels, -1)

    images = generator([noise, labels], training=False)
    images = (images + 1) / 2.0

    plt.figure(figsize=(10, 4))
    for i in range(26):
        plt.subplot(3, 9, i+1)
        plt.imshow(images[i, :, :, 0], cmap='gray')
        plt.title(chr(i + 65))
        plt.axis('off')
    plt.tight_layout()
    plt.show()

for epoch in range(EPOCHS):
    for images, labels in dataset:
        g_loss, d_loss = train_step(images, labels)

    print(f"Epoch {epoch+1}/{EPOCHS} | D: {d_loss:.4f} | G: {g_loss:.4f}")

    if (epoch + 1) % 5 == 0:
        generate_samples(epoch)