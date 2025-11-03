import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_channels):
        super(Generator, self).__init__()
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.init_size = 7
        self.fc = nn.Linear(latent_dim + num_classes, 256 * self.init_size * self.init_size)
        self.convblock = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.img_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        labels_onehot = F.one_hot(labels, self.num_classes).float()
        gen_input = torch.cat([z, labels_onehot], dim=1)
        x = self.fc(gen_input)
        x = x.view(-1, 256, self.init_size, self.init_size)
        img = self.convblock(x)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_size, num_classes):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)
        self.convblock = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_emb = self.label_embedding(labels).view(-1, 1, self.img_size, self.img_size)
        x = torch.cat([img, label_emb], dim=1)
        x = self.convblock(x)
        x = self.fc(x)
        return x