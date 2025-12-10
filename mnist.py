import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# =====================================================
# 1. 数据加载与预处理
# =====================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# =====================================================
# 2. 构建 CNN 模型
# =====================================================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


model = CNN()


# =====================================================
# 3. 损失函数 & 优化器
# =====================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# =====================================================
# 4. 训练
# =====================================================
loss_history = []
epochs = 5

print("Start Training...\n")

for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loss_history.append(loss.item())

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(trainloader):.4f}")

print("\nTraining Finished!\n")


# =====================================================
# 5. 测试模型
# =====================================================
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in testloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%\n")


# =====================================================
# 6. 可视化：训练 Loss 曲线
# =====================================================
plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.title("Training Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


# =====================================================
# 7. 可视化：展示预测结果
# =====================================================
images, labels = next(iter(testloader))
outputs = model(images)
_, preds = torch.max(outputs, 1)

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i].squeeze(), cmap='gray')
    plt.title(f"P:{preds[i].item()} / T:{labels[i].item()}")
    plt.axis("off")

plt.show()


# =====================================================
# 8. 保存模型（作业可选）
# =====================================================
torch.save(model.state_dict(), "mnist_cnn.pth")
print("Model saved as mnist_cnn.pth")
