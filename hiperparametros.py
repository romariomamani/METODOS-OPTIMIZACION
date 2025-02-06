import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Definir transformaciones para normalizar los datos
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Cargar dataset FashionMNIST
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Hiperparámetros a probar
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [32, 64, 128]

# Definir red neuronal simple
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Función para entrenar el modelo
def train_model(lr, batch_size, epochs=5):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_loss = []
    val_acc = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss.append(running_loss / len(trainloader))
        
        # Evaluar en conjunto de prueba
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        val_acc.append(accuracy)
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(trainloader):.4f}, Validation Accuracy: {accuracy:.2f}%')
    
    return train_loss, val_acc

# Probar diferentes combinaciones de hiperparámetros
results = {}
for lr in learning_rates:
    for batch_size in batch_sizes:
        print(f"Entrenando con lr={lr}, batch_size={batch_size}")
        train_loss, val_acc = train_model(lr, batch_size)
        results[(lr, batch_size)] = (train_loss, val_acc)

# Graficar resultados
plt.figure(figsize=(12, 5))
for (lr, batch_size), (train_loss, val_acc) in results.items():
    plt.plot(val_acc, label=f'lr={lr}, batch_size={batch_size}')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.title('Comparación de Exactitud en Validación')
plt.show()
