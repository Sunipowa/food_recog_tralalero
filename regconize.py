import os
import json
import warnings
from collections import defaultdict

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Định nghĩa model
class CNNWithThreeHeads(nn.Module):
    def __init__(self):
        super(CNNWithThreeHeads, self).__init__()
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
        self.head1 = nn.Sequential(
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        self.head3 = nn.Sequential(
            nn.Linear(1 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        feat = self.backbone(x)
        flat_feat = self.flatten(feat)
        out1 = self.head1(flat_feat)
        avg_feat = self.avgpool(feat)
        out2 = self.head2(avg_feat)
        gray_feat = feat.mean(dim=1, keepdim=True)
        gray_flat = self.flatten(gray_feat)
        out3 = self.head3(gray_flat)
        return torch.cat([out1, out2, out3], dim=1)

# 2. Load model + class names
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("cnn_model/food_cnn_final.pth", map_location=device)
model.eval()

train_dir = "lan cuoi nka/lan cuoi nka/test"
class_names = datasets.ImageFolder(train_dir).classes

# 3. Transform test ảnh
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# 4. Dự đoán ảnh trong thư mục
def predict_folder(folder_path):
    results = defaultdict(int)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                _, pred = torch.max(output, 1)
                label = class_names[pred.item()]
                results[label] += 1
    return dict(results)

# 5. Tính tiền từ file JSON
def calculate_bill(predictions, price_json_path):
    with open(price_json_path, "r", encoding="utf-8") as f:
        price_list = json.load(f)

    total = 0
    print("BILL:")
    for item, qty in predictions.items():
        price = price_list.get(item, 0)
        subtotal = qty * price
        total += subtotal
        print(f"{item}: {qty} x {price:,} = {subtotal:,} VND")

    print(f"TOTAL: {total:,} VND")

# 6. Chạy
if __name__ == "__main__":
    folder_path = "cropped_bowls"           # Thư mục chứa ảnh
    price_path = "gia_tien.json"            # Bảng giá

    print(f"Predicting images in: {folder_path}")
    predictions = predict_folder(folder_path)

    if predictions:
        calculate_bill(predictions, price_path)
    else:
        print("Không tìm thấy ảnh hợp lệ trong thư mục.")
