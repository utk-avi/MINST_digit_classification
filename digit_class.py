import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, save, load
from torch import optim
from torch.optim import Adam


transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*22*22, 128)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
            
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = ImageClassifier().to(device)

optimizer = Adam(classifier.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epochs in range(10):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = classifier(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epochs+1}, Loss: {loss.item()}")

torch.save(classifier.state_dict(), 'model_state.pt')

with open('model_state.pt', 'rb') as f: 
    classifier.load_state_dict(load(f))

img = Image.open('img_9')
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img).unsqueeze(0).to(device)
output = classifier(img_tensor)
predicted_label = torch.argmax(output)
print(f"Predicted label: {predicted_label}")