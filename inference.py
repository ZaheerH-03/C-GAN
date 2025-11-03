import torch
import matplotlib.pyplot as plt
from model import Generator


LATENT_DIM = 100
IMG_SIZE = 28
NUM_CLASSES = 10
IMG_CHANNELS = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Load Generator ----
G_model = Generator(LATENT_DIM, NUM_CLASSES, IMG_CHANNELS).to(device)
G_model.load_state_dict(torch.load("generator_weights.pth", map_location=device))
G_model.eval()

# ---- Generate images ----
label = torch.full((16,), 6, dtype=torch.long, device=device)  # label = digit 6
z = torch.randn(16, LATENT_DIM).to(device)

with torch.no_grad():
    fake_images = G_model(z, label)

# ---- Normalize to [0,1] range ----
fake_images = (fake_images + 1) / 2.0

# ---- Show one image ----
plt.imshow(fake_images[0].cpu().squeeze(), cmap='gray')
plt.title("Generated Digit 6")
plt.axis("off")
plt.show()