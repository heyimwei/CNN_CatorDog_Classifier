import os
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
from model import SimpleCNN

# 載入類別名稱
with open("classes.json") as f:
    classes = json.load(f)

# 載入模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# 影像轉換
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 推論函式
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()
    return classes[pred]

# 評估 val 資料
y_true, y_pred = [], []
val_root = "data/val"

for class_name in os.listdir(val_root):
    class_dir = os.path.join(val_root, class_name)
    if not os.path.isdir(class_dir): continue

    for filename in os.listdir(class_dir):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        filepath = os.path.join(class_dir, filename)
        pred = predict(filepath)
        y_pred.append(pred)
        y_true.append(class_name)

# 準確率
correct = sum(p == t for p, t in zip(y_pred, y_true))
accuracy = correct / len(y_true)
print(f"總共圖片數量：{len(y_true)}，正確數量：{correct}，準確率：{accuracy:.2%}")

# 混淆矩陣
cm = confusion_matrix(y_true, y_pred, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title(f"Accuracy: {accuracy:.2%}")
plt.tight_layout()
plt.show()
