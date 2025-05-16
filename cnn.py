import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Dùng GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. Transform ảnh
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# 3. Load dữ liệu train/test
train_dir = 'lan cuoi nka/lan cuoi nka/train'
test_dir = 'lan cuoi nka/lan cuoi nka/test'

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4. CNN model với 3 head (head3 nhận grayscale feature map)
import torch
import torch.nn as nn

class CNNWithThreeHeads(nn.Module):
    def __init__(self):
        super(CNNWithThreeHeads, self).__init__()

        # Feature extractor chung (backbone)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()

        # Head 1: Học đặc trưng tổng thể (global features)
        self.head1 = nn.Sequential(
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        # Head 2: Học đặc trưng trung bình không gian (spatial pooled)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # (128,1,1)
        self.head2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

        # Head 3: Học từ ảnh grayscale (thông tin edge, ánh sáng)
        self.head3 = nn.Sequential(
            nn.Linear(1 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.backbone:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        for head in [self.head1, self.head2, self.head3]:
            for m in head:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.backbone(x)  # output (B,128,8,8)

        # Head 1: Global features
        flat_feat = self.flatten(feat)
        out1 = self.head1(flat_feat)

        # Head 2: Spatial-pooled features
        avg_feat = self.avgpool(feat)  # (B,128,1,1)
        out2 = self.head2(avg_feat)

        # Head 3: Grayscale input features
        gray_feat = feat.mean(dim=1, keepdim=True)  # (B,1,8,8)
        gray_flat = self.flatten(gray_feat)
        out3 = self.head3(gray_flat)

        # Kết hợp 3 head lại
        return torch.cat([out1, out2, out3], dim=1)  # (B, 15)



# 5. Khởi tạo model
model = CNNWithThreeHeads().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 6. Warm-up scheduler
warmup_steps = len(train_loader) * 5  # warm-up trong 5 epoch

def warmup_lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)

# 7. Training
num_epochs = 100
loss_list = []
acc_list = []
lr_list = []

global_step = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        global_step += 1
        scheduler.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        lr_list.append(scheduler.get_last_lr()[0])

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    loss_list.append(epoch_loss)
    acc_list.append(epoch_acc)

    print(f"[Epoch {epoch+1:03}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

# 8. Testing
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\n📊 Final Test Accuracy: {100 * correct / total:.2f}%")

# 9. Lưu model
torch.save(model, "food_cnn_final.pth")
print("Model saved as food_cnn_multiheads.pth")

# 10. Lưu hình ảnh trực quan hóa
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(loss_list, label='Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(acc_list, label='Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy per Epoch')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(lr_list, label='LR', color='blue')
plt.xlabel('Batch step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Warm-up')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("training_metrics.png")
print("Đã lưu biểu đồ thành 'training_metrics.png'")
