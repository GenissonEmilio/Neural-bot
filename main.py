import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Configuração de Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Hiperparâmetros
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
PATCH_SIZE = 4
IMG_SIZE = 32
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 4
NUM_CLASSES = 10


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=64, num_heads=4, num_layers=4, num_classes=10):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2

        # Patch Embedding
        self.patch_embed = PatchEmbedding(3, patch_size, embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.num_patches, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cabeçalho de Classificação (MLP Head)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed

        x = self.transformer(x)

        cls_output = x[:, 0]

        return self.mlp_head(cls_output)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Bloco 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Bloco 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Bloco 3
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def train_model(model, train_loader, optimizer, criterion, epochs, name="Model"):
    model.train()
    history = []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"{name} Epoch {epoch + 1}/{epochs}", leave=False)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item())

        acc = 100 * correct / total
        print(f"[{name}] Epoch {epoch + 1}: Loss = {total_loss / len(train_loader):.4f}, Acc = {acc:.2f}%")
        history.append(acc)

    return history


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

vit_model = VisionTransformer().to(device)
cnn_model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
vit_optimizer = optim.Adam(vit_model.parameters(), lr=LEARNING_RATE)
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)

print("\n--- Treinando Vision Transformer ---")
vit_history = train_model(vit_model, trainloader, vit_optimizer, criterion, EPOCHS, "ViT")

print("\n--- Treinando CNN Tradicional ---")
cnn_history = train_model(cnn_model, trainloader, cnn_optimizer, criterion, EPOCHS, "CNN")

plt.figure(figsize=(10, 5))
plt.plot(vit_history, label='ViT Accuracy', marker='o')
plt.plot(cnn_history, label='CNN Accuracy', marker='x')
plt.title('Comparação de Performance: ViT vs CNN (CIFAR-10)')
plt.xlabel('Épocas')
plt.ylabel('Acurácia (%)')
plt.legend()
plt.grid(True)
plt.savefig('comparacao_vit_cnn.png')
plt.show()

print("\nProjeto concluído! Gráfico salvo como 'comparacao_vit_cnn.png'")