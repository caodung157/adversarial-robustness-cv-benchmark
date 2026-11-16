import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchattacks
import matplotlib.pyplot as plt

# 1. Cấu hình thiết bị (Tự động dùng GPU NVIDIA nếu có, không thì dùng CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang chạy thí nghiệm trên thiết bị: {device}")

# 2. Tải dữ liệu FashionMNIST (Ảnh xám 28x28, 10 lớp trang phục)
# Lưu ý: torchattacks yêu cầu dữ liệu pixel nằm trong khoảng [0.0, 1.0]
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

classes = [
    "Áo thun", "Quần dài", "Áo len", "Váy", "Áo khoác",
    "Sandal", "Áo sơ mi", "Giày Sneaker", "Túi xách", "Bốt cổ cao"
]

# 3. Xây dựng kiến trúc mạng CNN phân loại ảnh cơ bản
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 4. Huấn luyện mô hình trên dữ liệu sạch (2 epochs)
print("\n--- Giai đoạn 1: Huấn luyện mô hình cơ sở ---")
model.train()
for epoch in range(2):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/2 - Loss: {total_loss/len(train_loader):.4f}")

# 5. Đánh giá độ chính xác trên dữ liệu sạch (Clean Accuracy)
model.eval()
correct_clean = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct_clean += (predicted == labels).sum().item()

clean_acc = 100 * correct_clean / total
print(f"\n[Kết quả] Độ chính xác trên ảnh sạch (Clean Accuracy): {clean_acc:.2f}%")

# 6. Thực hiện tấn công đối kháng PGD (Projected Gradient Descent)
# eps=0.15 là ngân sách nhiễu tối đa trên từng điểm ảnh
print("\n--- Giai đoạn 2: Chạy tấn công đối kháng PGD ---")
atk = torchattacks.PGD(model, eps=0.15, alpha=0.02, steps=10)

correct_adv = 0
sample_clean, sample_adv, sample_true, sample_pred = None, None, None, None

for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    # Tạo ảnh nhiễu đối kháng
    adv_images = atk(images, labels)

    with torch.no_grad():
        outputs = model(adv_images)
        _, predicted = torch.max(outputs, 1)
        correct_adv += (predicted == labels).sum().item()

    # Lưu lại 1 mẫu bị đánh lừa thành công để xuất biểu đồ cho GitHub README
    if sample_clean is None:
        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                sample_clean = images[i].cpu()
                sample_adv = adv_images[i].cpu()
                sample_true = labels[i].item()
                sample_pred = predicted[i].item()
                break

adv_acc = 100 * correct_adv / total
print(f"[Kết quả] Độ chính xác sau tấn công PGD (Adversarial Accuracy): {adv_acc:.2f}%")

# 7. Trực quan hóa Ảnh gốc - Ma trận nhiễu - Ảnh bị tấn công
if sample_clean is not None:
    os.makedirs("assets", exist_ok=True)
    noise = (sample_adv - sample_clean).squeeze().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(sample_clean.squeeze(), cmap="gray")
    axes[0].set_title(f"Ảnh gốc\nNhãn thật: {classes[sample_true]}")
    axes[0].axis("off")

    axes[1].imshow(noise, cmap="bwr", vmin=-0.15, vmax=0.15)
    axes[1].set_title("Ma trận nhiễu (Delta)\n(Bản đồ điểm ảnh bị sửa)")
    axes[1].axis("off")

    axes[2].imshow(sample_adv.squeeze(), cmap="gray")
    axes[2].set_title(f"Ảnh đối kháng (PGD)\nAI đoán nhầm: {classes[sample_pred]}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("assets/pgd_demo.png", dpi=150)
    print("\nĐã lưu biểu đồ trực quan tại: assets/pgd_demo.png")