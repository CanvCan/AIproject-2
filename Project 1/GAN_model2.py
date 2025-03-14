import torch.optim as optim
import torch.nn as nn
import torchvision
import torch
import os

from torchvision import datasets, transforms
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = datasets.ImageFolder(root="C:\\Users\\Can\\Desktop\\Pigs Dataset (924 images version)",
                                     transform=transform)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

os.makedirs("images", exist_ok=True)


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.78),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.78),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(64, momentum=0.82),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.82),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


def train(num_epochs, latent_dim, optimizer_d, optimizer_g, adversarial_loss, generator, discriminator):
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(dataloader):

            valid = torch.ones(images.size(0), 1, device=device)
            fake = torch.zeros(images.size(0), 1, device=device)

            images = images.to(device)

            # Train Discriminator

            optimizer_d.zero_grad()
            z = torch.randn(images.size(0), latent_dim, device=device)
            fake_images = generator(z)

            real_loss = adversarial_loss(discriminator(images), valid)
            fake_loss = adversarial_loss(discriminator(fake_images.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_d.step()

            # Train Generator

            optimizer_g.zero_grad()
            gen_images = generator(z)

            g_loss = adversarial_loss(discriminator(gen_images), valid)
            g_loss.backward()
            optimizer_g.step()

            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] Batch {i + 1}/{len(dataloader)} "
                    f"Discriminator Loss: {d_loss.item():.4f} "
                    f"Generator Loss: {g_loss.item():.4f}"
                )

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                z = torch.randn(16, latent_dim, device=device)
                generated = generator(z).detach().cpu()
                grid = torchvision.utils.make_grid(generated, nrow=4, normalize=True)
                # Save the generated images
                save_image(grid, f"images10/epoch_{epoch + 1}.png")


def main():
    latent_dim = 100
    learning_rate = 0.0002
    num_epochs = 1000

    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    adversarial_loss = nn.BCELoss()

    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

    train(num_epochs, latent_dim, optimizer_d, optimizer_g, adversarial_loss, generator, discriminator)


main()
