import os
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load the trained generator model
generator = load_model('generator_model.h5')

# Define the latent dimension
latent_dim = 100
 
def generate_and_save_images(generator, latent_dim, num_images=5, save_path='generated_images'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    noise = np.random.normal(0, 1, (num_images, latent_dim))
    generated_images = generator.predict(noise)

    # Rescale images 0 - 1
    generated_images = 0.5 * generated_images + 0.5

    for i, img in enumerate(generated_images):
        plt.figure()
        if len(img.shape) == 3:  # For images with channels (e.g., RGB)
            plt.imshow(img)
        elif len(img.shape) == 2:  # For grayscale images
            plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(save_path, f'image_{i}.png'))
        plt.close()

# Generate and save new images
generate_and_save_images(generator, latent_dim)