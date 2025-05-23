import torch
from PIL import Image
from torchvision import transforms
from model import SimpleCNN

# 手動定義類別（依照你資料集目錄）
class_names = ['cats', 'dogs']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = SimpleCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

def predict(image_path):
    img = Image.open(image_path).convert('RGB')
    x   = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        idx    = logits.argmax(dim=1).item()
    return class_names[idx]

# 測試用圖片路徑
print(predict('data/val/cats/cat.4013.jpg'))
print(predict('data/val/cats/cat.4021.jpg'))
print(predict('data/val/cats/cat.4017.jpg'))
