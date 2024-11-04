import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Normalize the data
data = (data - 127.5) / 127.5

# Determine the shape of the images in your dataset
img_shape = (data.shape[1], 1)  # The data is now in a flattened format (data points for each landmark)
print("Shape of each image in dataset:", img_shape)

# Define GAN parameters
latent_dim = 100

# Build Generator
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    model.summary()

    noise = tf.keras.Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)

# Build Discriminator
def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = tf.keras.Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

# Compile models
optimizer = Adam(0.0002, 0.5)

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Build the generator
generator = build_generator()

# The generator takes noise as input and generates images
z = tf.keras.Input(shape=(latent_dim,))
img = generator(z)

# For the combined model, we will only train the generator
discriminator.trainable = False

# The discriminator takes generated images as input and determines validity
validity = discriminator(img)

# The combined model (stacked generator and discriminator)
# Trains the generator to fool the discriminator
combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

# Training the GAN
def train(epochs, batch_size=128, save_interval=50):

    # Load and rescale data
    X_train = data

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        # Sample noise as generator input
        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # Generate a batch of new images
        gen_imgs = generator.predict(noise)

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator (to have the discriminator label samples as valid)
        g_loss = combined.train_on_batch(noise, valid)

        # Print the progress
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")

        # If at save interval, save generated image samples
        if epoch % save_interval == 0:
            save_imgs(epoch)

# Function to save generated images
def save_imgs(epoch):
    noise = np.random.normal(0, 1, (5, latent_dim))
    gen_imgs = generator.predict(noise)
    
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    for i, img in enumerate(gen_imgs):
        plt.figure()
        if len(img_shape) == 3:  # For images with channels (e.g., RGB)
            plt.imshow(img)
        elif len(img_shape) == 2:  # For grayscale images
            plt.imshow(img.reshape(img_shape[0], img_shape[1]), cmap='gray')
        plt.savefig(f"images/{epoch}_{i}.png")
        plt.close()

# Create folder to save images
if not os.path.exists('images'):
    os.makedirs('images')

# Train the GAN
train(epochs=10000, batch_size=32, save_interval=200)

# Save the generator model
generator.save('generator_model.h5')