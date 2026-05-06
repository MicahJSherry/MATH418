import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

import os
from datetime import datetime
# -----------------------
# EMNIST LABEL MAP (47 classes)
# -----------------------
EMNIST_LABELS = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]

# -----------------------
# Load EMNIST Balanced
# -----------------------
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
# Hyperparameters
# --------------------
BATCH_SIZE = 128
LATENT_DIM = 100
NUM_CLASSES = 47
EPOCHS = 100
IMG_SHAPE = (28, 28, 1)

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

    label_embedding = layers.Embedding(NUM_CLASSES, 64)(label)
    label_embedding = layers.Flatten()(label_embedding)

    x = layers.Concatenate()([noise, label_embedding])

    x = layers.Dense(7 * 7 * 128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Reshape((7, 7, 128))(x)

    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    img = layers.Conv2D(1, 3, padding='same', activation='tanh')(x)

    return tf.keras.Model([noise, label], img)

# -----------------------
# Discriminator
# -----------------------
def build_discriminator():
    img = layers.Input(shape=IMG_SHAPE)
    label = layers.Input(shape=(1,), dtype='int32')

    label_embedding = layers.Embedding(NUM_CLASSES, 28 * 28)(label)
    label_embedding = layers.Reshape((28, 28, 1))(label_embedding)

    x = layers.Concatenate()([img, label_embedding])

    x = layers.Conv2D(64, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model([img, label], x)

# -----------------------
# Build models
# -----------------------
generator = build_generator()
discriminator = build_discriminator()

#plot_model(generator, to_file='gan_generator2.png', show_shapes=True, show_layer_names=True)
#plot_model(discriminator, to_file='gan_discriminator2.png', show_shapes=True, show_layer_names=True)
#exit()


# -----------------------
# Loss + Optimizers
# -----------------------
loss_fn = tf.keras.losses.BinaryCrossentropy()

g_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

# -----------------------
# Training step
# -----------------------
@tf.function
def train_step(real_images, labels):

    batch_size = tf.shape(real_images)[0]

    real = tf.ones((batch_size, 1)) * 0.9
    fake = tf.zeros((batch_size, 1))

    noise = tf.random.normal((batch_size, LATENT_DIM))
    random_labels = tf.random.uniform((batch_size, 1), 0, NUM_CLASSES, dtype=tf.int32)

    # ---- Generator ----
    with tf.GradientTape() as tape:
        gen_images = generator([noise, random_labels], training=True)
        pred = discriminator([gen_images, random_labels], training=True)
        g_loss = loss_fn(real, pred)

    grads = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    # ---- Discriminator ----
    with tf.GradientTape() as tape:
        real_pred = discriminator([real_images, tf.expand_dims(labels, -1)], training=True)
        fake_pred = discriminator([gen_images, random_labels], training=True)

        d_loss = (
            loss_fn(real, real_pred) +
            loss_fn(fake, fake_pred)
        ) / 2

    grads = tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

    return g_loss, d_loss

# -----------------------
# Sampling
# -----------------------
def generate_samples(epoch):

    noise = tf.random.normal((NUM_CLASSES, LATENT_DIM))
    labels = tf.range(0, NUM_CLASSES, dtype=tf.int32)
    labels = tf.expand_dims(labels, -1)

    imgs = generator([noise, labels], training=False)
    imgs = (imgs + 1) / 2.0

    plt.figure(figsize=(12, 8))

    for i in range(NUM_CLASSES):
        plt.subplot(6, 8, i + 1)
        plt.imshow(imgs[i, :, :, 0], cmap='gray')
        plt.title(EMNIST_LABELS[i], fontsize=6)
        plt.axis('off')

    plt.tight_layout()

    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/epoch_{epoch+1}.png")
    plt.close()



# Get current date and time
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

os.makedirs(f"models/{timestamp}", exist_ok=True)
# -----------------------
# Training loop
# -----------------------
for epoch in range(EPOCHS):

    for real_images, labels in dataset:
        g_loss, d_loss = train_step(real_images, labels)

    print(f"Epoch {epoch+1}/{EPOCHS} | D: {d_loss:.4f} | G: {g_loss:.4f}")

    if (epoch + 1) % 5 == 0:
        generate_samples(epoch)
        generator.save(f"models/{timestamp}/generator_{epoch+1}.keras")
        discriminator.save(f"models/{timestamp}/discriminator_{epoch+1}.keras")
        