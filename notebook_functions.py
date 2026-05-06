import tensorflow as tf
import matplotlib.pyplot as plt

# general imports 
import os
import re
import math

EMNIST_LABELS = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]

def load_generators(directory, prefix="generator_", suffix=".keras"): 
    

    files = [f for f in os.listdir(directory) if f.startswith(prefix)]
    
    def _get_epoch(name: str) -> int:
        match = re.search(rf"{re.escape(prefix)}(\d+)", name)
        if not match:
            raise ValueError(f"No number found in: {name}")
        return int(match.group(1))
    
    files = sorted(files, key=lambda f: _get_epoch(f))        
    models = []

    for f in files:
            
        models.append(tf.keras.models.load_model(f"{directory}/{f}"))

    return models





def generate_label_progression(label, generators, epoch_indices=None, LATENT_DIM=100, cols=5, label_titles=EMNIST_LABELS):

    title = label_titles[label]    
    num_models = len(generators)
    rows = math.ceil(num_models / cols)

    noise = tf.random.normal((num_models, LATENT_DIM))

    labels = tf.constant([label] * num_models, dtype=tf.int32)
    labels = tf.expand_dims(labels, -1)

    plt.figure(figsize=(cols * 2, rows * 2.5))

    plt.suptitle(f"{title} Progression", fontsize=14)

    for i, gen in enumerate(generators):
        img = gen([noise[i:i+1], labels[i:i+1]], training=False)
        img = (img + 1) / 2.0

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img[0, :, :, 0], cmap='gray')

        if epoch_indices:
            plt.title(f"Epoch {epoch_indices[i]}", fontsize=8)

        plt.axis('off')

    plt.tight_layout()

    os.makedirs("images/progress", exist_ok=True)
    plt.savefig(f"images/progress/label_{title}_progression.png")
    plt.close()




"""generators = load_generators("models/2026-05-01_15-21-27")

for i in range(len(EMNIST_LABELS)):
    generate_label_progression(i, generators, epoch_indices=list(range(5, 101, 5)) )"""




def plot_generated_samples(generator, label, num_samples=16, latent_dim=100, cols=4):
    """
    Generates and plots multiple images for a given label using a GAN generator.
    
    generator: trained generator model
    label: int class label
    num_samples: number of images to generate
    latent_dim: size of noise vector
    cols: number of columns in grid
    title: optional plot title
    """
    title = EMNIST_LABELS[label]
    rows = math.ceil(num_samples / cols)

    noise = tf.random.normal((num_samples, latent_dim))
    labels = tf.constant([label] * num_samples, dtype=tf.int32)
    labels = tf.expand_dims(labels, -1)

    imgs = generator([noise, labels], training=False)
    imgs = (imgs + 1) / 2.0  # rescale from [-1, 1] to [0, 1]

    plt.figure(figsize=(cols * 2, rows * 2))

    plt.suptitle(f"Generated Samples - Label {title}", fontsize=14)
    
    
    for i in range(num_samples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(imgs[i, :, :, 0], cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"images/noise/noise_{title}.png")

generator = tf.keras.models.load_model(f"models/2026-05-01_15-21-27/generator_100.keras")

for i in range(len(EMNIST_LABELS)):
    plot_generated_samples(generator, i)