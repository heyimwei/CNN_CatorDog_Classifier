from kaggle.api.kaggle_api_extended import KaggleApi
import os, zipfile, shutil

# 參數
DATASET_SLUG = 'tongpython/cat-and-dog'  # 或 adityakadiyal/dogs-vs-cats
ZIP_NAME     = 'cat-and-dog.zip'
EXTRACT_DIR  = 'raw_data'
OUTPUT_DIR   = 'data'
NUM_PER_CLASS = 1000

# 1. 下載並解壓
api = KaggleApi(); api.authenticate()
api.dataset_download_files(DATASET_SLUG, path='.', unzip=False)

with zipfile.ZipFile(ZIP_NAME, 'r') as zf:
    zf.extractall(EXTRACT_DIR)

# 2. 建立資料夾
for split in ['train','val']:
    for cls in ['cats','dogs']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# 3. 複製前一千張到 train，再一千到 val
def copy_images(src_folder, prefix, start, count, dst_folder):
    imgs = sorted(f for f in os.listdir(src_folder) if f.startswith(prefix))
    for fname in imgs[start:start+count]:
        shutil.copy(os.path.join(src_folder, fname),
                    os.path.join(dst_folder, fname))

SRC = os.path.join(EXTRACT_DIR, 'PetImages')  # 視此 dataset 結構而定
# 若 PetImages 底下有 Cats/ Dogs/，可調整 prefix 和路徑
copy_images(os.path.join(SRC,'Cat'), 'Cat', 0, NUM_PER_CLASS, os.path.join(OUTPUT_DIR,'train','cats'))
copy_images(os.path.join(SRC,'Cat'), 'Cat', NUM_PER_CLASS, NUM_PER_CLASS, os.path.join(OUTPUT_DIR,'val','cats'))
copy_images(os.path.join(SRC,'Dog'), 'Dog', 0, NUM_PER_CLASS, os.path.join(OUTPUT_DIR,'train','dogs'))
copy_images(os.path.join(SRC,'Dog'), 'Dog', NUM_PER_CLASS, NUM_PER_CLASS, os.path.join(OUTPUT_DIR,'val','dogs'))

print("資料整理完成！")
