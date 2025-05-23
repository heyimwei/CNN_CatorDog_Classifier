# CNN 🐶/🐱? Classifier 

一個基於 PyTorch 的簡易小動物圖片分類系統，並提供 Flask 前端介面供使用者上傳圖片並即時取得預測結果。

## 📁 專案結構

```
.
├── datasets.py           # 資料載入與前處理
├── model.py              # CNN 模型結構
├── train.py              # CNN 模型訓練
├── inference.py          # 推論邏輯封裝
├── app.py                # Flask 主程式
├── templates/
│   └── index.html        # 上傳圖片前端介面
├── data/               
│   └── train/            # 訓練集
│       └── cats/         # 貓
│       └── dogs/         # 狗
│   └── val/              # 驗證集
│       └── cats/         # 貓
│       └── dogs/         # 狗
├── .gitignore            # Git 忽略清單
├── requirements.txt      # 套件安裝清單
├── README.md             # 本說明文件
└── screenshot/           # 網頁操作實例
````

## 🚀 使用方式

### 1️⃣ 安裝依賴套件
```bash
pip install -r requirements.txt
````

### 2️⃣ 訓練模型

請先準備好已分類的圖片資料夾，並執行訓練腳本（如 `train.py`）。

```bash
python train.py
```

模型訓練完成後會產出 `best_model.pth`。

### 3️⃣ 啟動本地伺服器

```bash
python app.py
```

預設開啟在：`http://127.0.0.1:10000/`，可透過網頁上傳圖片並查看分類結果。

---

## 🧠 模型說明

使用 PyTorch 自建簡易 CNN 模型，適合用於二分類（`cats`, `dogs`），若要延伸多分類可以擴增類別資料夾與模型輸出層設定。

---

## 執行實例

這是已進行過一次分類的畫面

![alt text](screenshot/image.png)

可以再次選擇檔案

![alt text](screenshot/image-1.png)

按下分類

![alt text](screenshot/image-2.png)

再次完成分類！

![alt text](screenshot/image-3.png)

也可以清空畫面回到原樣

![alt text](screenshot/image-4.png)