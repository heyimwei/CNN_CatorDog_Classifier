import torch, os
from torch import optim, nn
from datasets import train_loader, val_loader, train_ds
from model    import SimpleCNN
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_acc = 0.0
save_path = 'best_model.pth'

for epoch in range(1,11):
    model.train()
    running_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # 驗證
    model.eval()
    correct, total = 0,0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds==labels).sum().item()
            total   += labels.size(0)
    acc = correct/total
    print(f"[Epoch {epoch}] loss={running_loss/len(train_loader):.4f}, val_acc={acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), save_path)
        print(f"  → Saved best model (acc={best_acc:.4f})")

with open("classes.json", "w") as f:
    json.dump(train_ds.classes, f)