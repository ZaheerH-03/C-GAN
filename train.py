import torch
import torchvision
import torch.optim as optim
from torchvision import transforms , datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import Generator,Discriminator
import torch.nn as nn
# HYPERPARAMETERS
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
LATENT_DIM = 100
NUM_CLASSES = 10
IMG_CHANNELS = 1
IMG_SIZE = 28
NUM_EPOCHS = 50

def train_cgan():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize models
    G = Generator(LATENT_DIM, NUM_CLASSES, IMG_CHANNELS).to(device)
    D = Discriminator(IMG_SIZE, NUM_CLASSES).to(device)

    # Optimizers
    optim_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optim_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    for epoch in range(NUM_EPOCHS):
        for i, (real_images, labels) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            labels = labels.to(device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            optim_D.zero_grad()
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            real_output = D(real_images, labels)
            d_real_loss = criterion(real_output, real_labels)

            z = torch.randn(batch_size, LATENT_DIM, device=device)
            random_labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
            fake_images = G(z, random_labels).detach()
            fake_output = D(fake_images, random_labels)
            d_fake_loss = criterion(fake_output, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optim_D.step()

            # ---------------------
            # Train Generator
            # ---------------------
            optim_G.zero_grad()
            z = torch.randn(batch_size, LATENT_DIM, device=device)
            random_labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
            gen_images = G(z, random_labels)
            g_output = D(gen_images, random_labels)
            g_loss = criterion(g_output, real_labels)  # Generator wants D to predict 1
            g_loss.backward()
            optim_G.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch {i}/{len(dataloader)} | "
                      f"D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        # ---------------------
        # Visualization
        # ---------------------
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                z_vis = torch.randn(16, LATENT_DIM, device=device)
                labels_vis = torch.randint(0, NUM_CLASSES, (16,), device=device)
                fake_samples = G(z_vis, labels_vis).view(-1, 1, 28, 28)
                plt.figure(figsize=(4,4))
                for idx in range(16):
                    plt.subplot(4, 4, idx+1)
                    plt.imshow(fake_samples[idx].cpu().squeeze(), cmap='gray')
                    plt.axis('off')
                plt.show()

    return G, D

G_model, D_model = train_cgan()
torch.save(G_model.state_dict(), "weights/generator_weights.pth")
torch.save(D_model.state_dict(), "weights/discriminator_weights.pth")