import os, shutil, random

# 來源資料夾
SRC_TRAIN_CATS = 'raw_data/training_set/cats'
SRC_TRAIN_DOGS = 'raw_data/training_set/dogs'
SRC_TEST_CATS  = 'raw_data/test_set/cats'
SRC_TEST_DOGS  = 'raw_data/test_set/dogs'

# 目標資料夾
DEST = 'data'
DEST_TRAIN_CATS = os.path.join(DEST, 'train', 'cats')
DEST_TRAIN_DOGS = os.path.join(DEST, 'train', 'dogs')
DEST_VAL_CATS   = os.path.join(DEST, 'val', 'cats')
DEST_VAL_DOGS   = os.path.join(DEST, 'val', 'dogs')

# 確保目標資料夾存在
for folder in [DEST_TRAIN_CATS, DEST_TRAIN_DOGS, DEST_VAL_CATS, DEST_VAL_DOGS]:
    os.makedirs(folder, exist_ok=True)

def copy_random_images(src_folder, dst_folder, count):
    all_imgs = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    selected = random.sample(all_imgs, count)
    for fname in selected:
        shutil.copy(os.path.join(src_folder, fname), os.path.join(dst_folder, fname))

# 複製 1000 張到 train
copy_random_images(SRC_TRAIN_CATS, DEST_TRAIN_CATS, 1000)
copy_random_images(SRC_TRAIN_DOGS, DEST_TRAIN_DOGS, 1000)

# 複製 1000 張到 val
copy_random_images(SRC_TEST_CATS, DEST_VAL_CATS, 1000)
copy_random_images(SRC_TEST_DOGS, DEST_VAL_DOGS, 1000)

print("✅ 成功複製各 1000 張到 data/train 與 data/val！")
