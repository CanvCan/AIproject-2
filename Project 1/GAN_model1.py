from keras.layers import Input, Dense, Reshape, Flatten, Conv2DTranspose, Conv2D, Dropout, UpSampling2D
from keras.layers import BatchNormalization, LeakyReLU
from keras.models import Sequential, Model
import tensorflow.keras.optimizers as tfk_opt
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


class GAN:
    def __init__(self, img_size=(64, 64), channels=3, latent_dim=100):
        self.img_rows = img_size[0]
        self.img_cols = img_size[1]
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=tfk_opt.Adam(0.002, 0.5),
                                   metrics=['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False

        validity = self.discriminator(img)

        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=tfk_opt.Adam(0.002, 0.5))

        os.makedirs("images", exist_ok=True)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(8 * 8 * 256, activation='relu', input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.78))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape(
            (8, 8, 256)))  # input: 8x8x256 (16384) boyutunda bir tensör. output: 8x8 boyutunda 256 kanallı bir tensör.
        model.add(Dropout(0.25))

        model.add(UpSampling2D())  # 8x8 -> 16x16
        model.add(Conv2D(128, 3, padding='same'))  # channel (256 --> 128)
        model.add(BatchNormalization(momentum=0.78))
        model.add(LeakyReLU(alpha=0.2))

        model.add(UpSampling2D())  # 16x16 -> 32x32
        model.add(Conv2D(64, 3, padding='same'))  # channel (128 --> 64)
        model.add(BatchNormalization(momentum=0.78))
        model.add(LeakyReLU(alpha=0.2))

        model.add(UpSampling2D())  # 32x32 -> 64x64
        model.add(Conv2D(32, 3, padding='same'))  # channel (64 --> 32)
        model.add(BatchNormalization(momentum=0.78))
        model.add(LeakyReLU(alpha=0.2))

        # input: 64x64 boyutunda ve 32 kanallı bir tensör.
        # output: 64x64 boyutunda ve "self.channels" kanallı bir tensör.
        model.add(Conv2DTranspose(self.channels, 3, padding='same', activation='tanh'))

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        # input: self.img_shape boyutunda bir görüntü (örneğin, 64x64x3).
        # output: 32 filtreyle (32 adet 3x3 filtre), boyutu yarıya indirgenerek (strides=2),
        # padding="same" ile boyutu korunarak (32x32x32).
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
        model.add(BatchNormalization(momentum=0.82))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        # input: (32x32x32)
        model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.82))
        model.add(LeakyReLU(alpha=0.25))
        model.add(Dropout(0.25))
        # output: (16x16x64)

        # input: (16x16x64)
        model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.82))
        model.add(LeakyReLU(alpha=0.25))
        model.add(Dropout(0.25))
        # output: (8x8x128)

        # input: (8x8x128)
        model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.82))
        model.add(LeakyReLU(alpha=0.25))
        model.add(Dropout(0.25))
        # output: (4x4x256)

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=5):
        x_train = load_images("C:\\Users\\Can\\Desktop\\Pigs Dataset (47 images version)",
                              img_size=(self.img_rows, self.img_cols))

        x_train = (x_train.astype(np.float32) - 127.5) / 127.5

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # Train the discriminator
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            images = x_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_images = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(images, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_images, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            g_loss = self.combined.train_on_batch(noise, valid)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_images = self.generator.predict(noise)

        gen_images = 0.5 * gen_images + 0.5

        fig, axs = plt.subplots(r, c, figsize=(10, 10))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_images[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


def load_images(directory, img_size=(64, 64)):
    images = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, img_size)
        images.append(image)
    return np.array(images)


if __name__ == '__main__':
    gan = GAN(img_size=(64, 64), channels=3, latent_dim=100)
    gan.train(epochs=10000, batch_size=16, sample_interval=200)
