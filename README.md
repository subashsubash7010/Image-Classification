# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Include the Problem Statement and Dataset.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Write your own steps

### STEP 2:

### STEP 3:


## PROGRAM

### Name:Subash M
### Register Number: 212224220109

```

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from torchsummary import summary

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=False, transform=transform, download=True
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f'Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}')


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier().to(device)

print('Name: Subassh.M')
print('Register Number: 212224220109')
summary(model, input_size=(1, 28, 28))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Name: Subash.M')
        print('Register Number: 212224220109')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

train_model(model, train_loader)

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print('Name: Subash.M')
    print('Register Number: 212224220109')
    print(f'Test Accuracy: {accuracy:.4f}')

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print('Name: Subash.M')
    print('Register Number: 212224220109')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

test_model(model, test_loader)


def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    class_names = dataset.classes
    print('Name: Subash.M')
    print('Register Number: 212224220109')
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}')

# Example: Predict image at index 80
predict_image(model, image_index=80, dataset=test_dataset)
```
## OUTPUT
### Training Loss per Epoch
<img width="829" height="559" alt="Screenshot 2026-02-18 102631" src="https://github.com/user-attachments/assets/f4956cfd-57ff-425e-b4f7-a49a8cf0c410" />


## Include the Training Loss per epoch
<img width="737" height="288" alt="Screenshot 2026-02-18 102906" src="https://github.com/user-attachments/assets/5873862c-e273-41c7-a5a4-898a491fbf69" />

### Confusion Matrix
<img width="814" height="677" alt="Screenshot 2026-02-18 103032" src="https://github.com/user-attachments/assets/6424602a-40f9-4ebf-95b4-073039a6e1f3" />


### Classification Report
<img width="935" height="438" alt="Screenshot 2026-02-18 103103" src="https://github.com/user-attachments/assets/56ad88d2-40ea-4041-9038-259ce8fba116" />


### New Sample Data Prediction
<img width="935" height="438" alt="Screenshot 2026-02-18 103103" src="https://github.com/user-attachments/assets/be2c1425-c3fd-4d3d-a45a-a88fd282ee47" />

## RESULT :
Thus, a convolutional deep neural network for image classification and to verify the response for new images is to developed successfully.


