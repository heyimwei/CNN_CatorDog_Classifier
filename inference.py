from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms
from model import SimpleCNN
from datasets import train_loader  # 取得 class 名稱

class Classifier:
    def __init__(self, model_path='best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = SimpleCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])

    def predict(self, img_bytes):
        try:
            img = Image.open(img_bytes).convert('RGB')
        except UnidentifiedImageError:
            raise ValueError('上傳的不是有效圖片檔案')

        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            idx = logits.argmax(dim=1).item()
        return train_loader.dataset.classes[idx]
