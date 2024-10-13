import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



# Set up the dataset and dataloader
transform = transforms.Compose([transforms.ToTensor()])
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(dataset=mnist_data, batch_size=64, shuffle=True)

# Initialize the model, loss function and optimizer
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for data in data_loader:
        img, _ = data
        img = img.view(img.size(0), -1)

        # Forward pass
        ouput = model(img)
        loss = criterion(ouput, img)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()\
        
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss:{loss.item():.4f}')

    print("Training finished!")


# Use the trained model
with torch.no_grad():
    # Get a batch of test images
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset=test_data, batch_size=10, shuffle=False)
    test_images, _ = next(iter(test_loader))

    # Flatten and reconstruct the test images
    test_images = test_images.view(test_images.size(0), -1)
    reconstructed = model(test_images)
    print(reconstructed)
    print(type(reconstructed))

    # Here you could visualize or save the original and reconstructed images
    print("Test images reconstructed!")