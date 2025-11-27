# =============================================================================
# STEP 2: Preprocessing with Train/Val/Test Split
# =============================================================================

import numpy as np
import SimpleITK as sitk
import os
import cc3d
import random
from skimage import transform
from tqdm import tqdm
import glob

# Set to True for a quick test (5 images), False for the full experiment (50 images)
LITE_MODE = False
# --- CONFIGURATION ---
nii_path = "/content/drive/MyDrive/Colab Notebooks/NN_Deep_Learning_Term-Project/FLARE22Train//images"
gt_path = "/content/drive/MyDrive/Colab Notebooks/NN_Deep_Learning_Term-Project/FLARE22Train//labels"

# Output Directories
base_npy_path = "data/npy/CT_Abd"
train_path = os.path.join(base_npy_path, "train")
val_path = os.path.join(base_npy_path, "val")
test_path = os.path.join(base_npy_path, "test")

for p in [train_path, val_path, test_path]:
    os.makedirs(os.path.join(p, "gts"), exist_ok=True)
    os.makedirs(os.path.join(p, "imgs"), exist_ok=True)

# Constants
image_size = 1024
voxel_num_thre2d = 100
voxel_num_thre3d = 1000
WINDOW_LEVEL = 40
WINDOW_WIDTH = 400

# --- LOAD & SPLIT ---
names = sorted(os.listdir(gt_path))
# Filter valid files
names = [n for n in names if os.path.exists(os.path.join(nii_path, n.replace('.nii.gz', '_0000.nii.gz')))]

if {LITE_MODE}:
    print("⚠️ LITE MODE: Using only 5 images total.")
    names = names[:50]

# Shuffle and Split
random.seed(42)
random.shuffle(names)

total = len(names)
n_train = int(total * 0.8)
n_val = int(total * 0.1)
# Ensure at least 1 image in each if dataset is tiny
if total < 10: n_train, n_val = total - 2, 1 

train_names = names[:n_train]
val_names = names[n_train:n_train+n_val]
test_names = names[n_train+n_val:]

print(f"Total Scans: {{total}}")
print(f"Split: Train={{len(train_names)}}, Val={{len(val_names)}}, Test={{len(test_names)}}")

# --- PROCESSING FUNCTION ---
def process_batch(file_list, save_root):
    for name in tqdm(file_list):
        image_name = name.replace('.nii.gz', '_0000.nii.gz')
        gt_sitk = sitk.ReadImage(os.path.join(gt_path, name))
        gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))
        gt_data_ori[gt_data_ori == 12] = 0 # remove duodenum

        # 3D Cleanup
        gt_data_ori = cc3d.dust(gt_data_ori, threshold=voxel_num_thre3d, connectivity=26, in_place=True)

        z_index, _, _ = np.where(gt_data_ori > 0)
        z_index = np.unique(z_index)

        if len(z_index) > 0:
            img_sitk = sitk.ReadImage(os.path.join(nii_path, image_name))
            image_data = sitk.GetArrayFromImage(img_sitk)
            
            # Windowing
            lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
            upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = ((image_data_pre - np.min(image_data_pre)) / (np.max(image_data_pre) - np.min(image_data_pre)) * 255.0)
            image_data_pre = np.uint8(image_data_pre)

            for i in z_index:
                # 2D Cleanup
                gt_slice = gt_data_ori[i, :, :]
                if np.sum(gt_slice) < voxel_num_thre2d: continue

                # Resize
                img_3c = np.repeat(image_data_pre[i, :, :, None], 3, axis=-1)
                resize_img = transform.resize(img_3c, (image_size, image_size), order=3, preserve_range=True, mode="constant", anti_aliasing=True)
                resize_img = (resize_img - resize_img.min()) / np.clip(resize_img.max() - resize_img.min(), a_min=1e-8, a_max=None)
                
                resize_gt = transform.resize(gt_slice, (256, 256), order=0, preserve_range=True, mode="constant", anti_aliasing=False)
                
                # Save
                save_name = name.split('.nii.gz')[0] + "-" + str(i).zfill(3) + ".npy"
                np.save(os.path.join(save_root, "imgs", save_name), resize_img)
                np.save(os.path.join(save_root, "gts", save_name), np.uint8(resize_gt))

print("Processing Training Set...")
process_batch(train_names, train_path)
print("Processing Validation Set...")
process_batch(val_names, val_path)
print("Processing Test Set...")
process_batch(test_names, test_path)
