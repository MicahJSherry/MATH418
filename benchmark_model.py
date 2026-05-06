import tensorflow as tf
import numpy as np
import os
from notebook_functions import load_generators
import pandas as pd 

import matplotlib.pyplot as plt
# -----------------------
# Settings
# -----------------------
NUM_CLASSES = 47
LATENT_DIM = 100
SAMPLES_PER_CLASS = 200  # adjust as needed


# -----------------------
# Load classifier
# -----------------------
classifier = tf.keras.models.load_model("models/emnist_classifier.keras")


# -----------------------
# Generate synthetic dataset
# -----------------------
def generate_synthetic_dataset(generator, samples_per_class=SAMPLES_PER_CLASS):
    all_images = []
    all_labels = []

    for label in range(NUM_CLASSES):

        noise = tf.random.normal((samples_per_class, LATENT_DIM))
        labels = tf.constant([label] * samples_per_class, dtype=tf.int32)
        labels = tf.expand_dims(labels, -1)

        imgs = generator([noise, labels], training=False)
        #imgs = (imgs + 1) / 2.0  # match classifier input scaling

        all_images.append(imgs.numpy())
        all_labels.append(np.array([label] * samples_per_class))

    X = np.concatenate(all_images, axis=0)
    y = np.concatenate(all_labels, axis=0)

    return X, y


# -----------------------
# Benchmark function
# -----------------------
def benchmark_generator(generator):
    class_accuracy = []
    print("Generating synthetic dataset...")
    X, y = generate_synthetic_dataset(generator)

    print("Running classifier predictions...")
    preds = classifier.predict(X, batch_size=128)
    preds = np.argmax(preds, axis=1)

    accuracy = np.mean(preds == y)
    print(f"\nOverall Accuracy on synthetic data: {accuracy:.4f}")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for c in range(NUM_CLASSES):
        idx = (y == c)
        if np.sum(idx) == 0:
            continue

        class_acc = np.mean(preds[idx] == y[idx])
        class_accuracy.append(class_acc) 
        print(f"Class {c}: {class_acc:.4f}")

    return accuracy, class_accuracy


# -----------------------
# Load generator(s)
# -----------------------


"""generators = load_generators("models/2026-05-01_15-21-27")

# -----------------------
# Run benchmark
# -----------------------
EMNIST_LABELS = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]



accuracy_by_epoch=[]
i = 5
for generator in generators:
    total_accuracy, class_accuracy = benchmark_generator(generator)
    accuracy_by_epoch.append([i, total_accuracy]+class_accuracy)
    i += 5

df = pd.DataFrame(accuracy_by_epoch,columns=["Epoch", "total_accuracy"]+EMNIST_LABELS)


df.to_csv("generator_benchmark.csv")"""


df = pd.read_csv("generator_benchmark.csv")

epochs = df["Epoch"]
accuracy = df["total_accuracy"]

plt.figure(figsize=(8, 5))

plt.plot(epochs, accuracy, marker="o", linewidth=2)

plt.title("GAN Generator Performance Over Time")
plt.xlabel("Epoch")
plt.ylabel("Overall Accuracy (Classifier on Synthetic Data)")

plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_by_epoch.png")