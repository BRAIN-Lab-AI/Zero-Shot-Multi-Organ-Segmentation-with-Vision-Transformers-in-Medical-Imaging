import os
import glob
import math
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from segment_anything import sam_model_registry
from monai.metrics import compute_dice, compute_surface_dice
from monai.networks.nets import BasicUNet  # Added for U-Net

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FLARE22 Test Data
TEST_DATA_DIR = "/content/drive/MyDrive/Colab Notebooks/NN_Deep_Learning_Term-Project/MedSAM/data/npy/CT_Abd/test"

# FLARE22 Parameters
FLARE22_SPACING = (0.644531, 0.644531, 2.5)
NSD_TOLERANCE = 2.0  # mm

# Model Checkpoints
SAM_CKPT = "work_dir/SAM/sam_vit_b_01ec64.pth"

# Updated to include U-Net and the two Baselines
MODEL_CONFIGS = {
    "U-Net-Specialist": {
        "checkpoint": "/content/drive/MyDrive/Colab Notebooks/NN_Deep_Learning_Term-Project/MedSAM/work_dir/50-60-NN_UNet_best_fixed.pth",
        "type": "unet",
        "frozen": False
    },
    "Baseline-Frozen": {
        "checkpoint": "/content/drive/MyDrive/Colab Notebooks/NN_Deep_Learning_Term-Project/MedSAM/work_dir/50-60-NN_MedSAM-Baseline_best.pth",
        "type": "baseline",
        "frozen": True
    },
    "Baseline-Unfrozen": {
        "checkpoint": "/content/drive/MyDrive/Colab Notebooks/NN_Deep_Learning_Term-Project/MedSAM/work_dir/Unfrozen_2_50-60-NN_MedSAM-Baseline_best.pth",
        "type": "baseline",
        "frozen": False
    }
}

# FLARE22 Organ Mapping
ORGAN_MAPPING = {
    "Liver": (1, 6),
    "Right Kidney": (2, 2),
    "Spleen": (3, 1),
    "Pancreas": (4, 10),
    "Left Kidney": (13, 3),
}

# Output directory
SAVE_DIR = "/content/drive/MyDrive/Colab Notebooks/NN_Deep_Learning_Term-Project/MedSAM/Subhannah-BaselineFLARE22_Final_Evaluation"
os.makedirs(SAVE_DIR, exist_ok=True)


# ==============================================================================
# MODEL ARCHITECTURES
# ==============================================================================

class MedSAM_Baseline(nn.Module):
    """Baseline MedSAM (no fusion, no LoRA)"""
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, box):
        emb = self.image_encoder(image)
        if box.dim() == 2:
            box = box.unsqueeze(1)  # [B, 1, 4]
        with torch.no_grad():
            sparse, dense = self.prompt_encoder(points=None, boxes=box, masks=None)
        masks, _ = self.mask_decoder(
            image_embeddings=emb,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False
        )
        return F.interpolate(masks, size=(1024, 1024), mode='bilinear', align_corners=False)

# ==============================================================================
# UTILITIES
# ==============================================================================

def get_bbox_from_mask(mask):
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0: return None
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    H, W = mask.shape
    x_min = max(0, x_min - 5)
    x_max = min(W, x_max + 5)
    y_min = max(0, y_min - 5)
    y_max = min(H, y_max + 5)
    return np.array([x_min, y_min, x_max, y_max])

def load_test_data(test_dir, organ_mapping):
    test_samples = []
    print(f"🔍 Loading FLARE22 test data from: {test_dir}")
    img_dir = os.path.join(test_dir, "imgs")
    gt_dir = os.path.join(test_dir, "gts")

    if not os.path.exists(img_dir):
        print(f"❌ Image directory not found: {img_dir}")
        return []

    all_files = sorted(glob.glob(os.path.join(img_dir, "*.npy")))

    for img_path in tqdm(all_files, desc="Indexing"):
        base_name = os.path.basename(img_path)
        gt_path = os.path.join(gt_dir, base_name)
        if not os.path.exists(gt_path): continue

        try:
            gt = np.load(gt_path)
            if np.max(gt) == 0: continue
            unique_labels = np.unique(gt)

            for organ_name, (flare_label, _) in organ_mapping.items():
                if flare_label in unique_labels:
                    gt_binary = (gt == flare_label).astype(np.uint8)
                    box = get_bbox_from_mask(gt_binary)
                    if box is None: continue

                    test_samples.append({
                        'image': np.load(img_path),
                        'gt': gt_binary,
                        'box': box[None, :],
                        'organ': organ_name,
                        'path': img_path,
                        'spacing': FLARE22_SPACING
                    })
        except Exception:
            continue
    return test_samples

def prepare_input(sample, model_type):
    """Prepare input for model"""
    img = sample['image']
    box = sample['box']

    if img.ndim == 3: img = img[:, :, 0]

    # Resize to 1024x1024
    original_h, original_w = img.shape[:2]
    target_size = 1024

    if original_h != target_size or original_w != target_size:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        scale_x = target_size / original_w
        scale_y = target_size / original_h
        box = box * np.array([scale_x, scale_y, scale_x, scale_y])

    # Normalize
    if img.max() > 1:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Stack channels
    img_stack = np.stack([img] * 3, axis=0)

    img_tensor = torch.from_numpy(img_stack).float().unsqueeze(0).to(DEVICE)
    if box.ndim == 2: box = box[0]
    box_tensor = torch.from_numpy(box).float().unsqueeze(0).to(DEVICE)

    return img_tensor, box_tensor

def compute_metrics(pred, gt, spacing):
    # Inputs: (1024, 1024) numpy arrays
    pred_t = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    gt_t = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)     # (1, 1, H, W)

    # 1. Dice
    try:
        dice = compute_dice(pred_t, gt_t).item() * 100.0
    except: dice = 0.0

    # 2. NSD (Normalized Surface Dice)
    # FIX: Use 2D calculation for slice-wise evaluation.
    # Previous code forced 3D on a single slice which can be unstable or undefined.
    try:
        # spacing is (0.64, 0.64, 2.5) -> we need (y, x) or (x, y) spacing for 2D.
        # Since x and y spacing are equal, we take the first two.
        spacing_2d = [spacing[0], spacing[1]]

        # compute_surface_dice handles (B, C, H, W) as 2D if spacing has length 2
        nsd = compute_surface_dice(
            pred_t,
            gt_t,
            class_thresholds=[NSD_TOLERANCE],
            spacing=spacing_2d
        ).item() * 100.0
    except Exception as e:
        # print(f"NSD Error: {e}") # Uncomment to debug
        nsd = 0.0

    return dice, nsd

# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_all_models(model_configs, sam_ckpt):
    models = {}
    print("\n🚀 Loading Models...")

    for name, cfg in model_configs.items():
        print(f"\n   Loading {name}...")
        try:
            if cfg['type'] == 'unet':
                model = BasicUNet(
                    spatial_dims=2,
                    in_channels=3,
                    out_channels=1,
                    features=(32, 32, 64, 128, 256, 32),
                    dropout=0.1,
                )
            else:
                sam = sam_model_registry["vit_b"](checkpoint=sam_ckpt)
                model = MedSAM_Baseline(sam.image_encoder, sam.mask_decoder, sam.prompt_encoder)

            if os.path.exists(cfg['checkpoint']):
                ckpt = torch.load(cfg['checkpoint'], map_location=DEVICE, weights_only=False)
                state_dict = ckpt.get('model', ckpt)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing: print(f"      ⚠️  Missing keys: {len(missing)}")
            else:
                print(f"      ⚠️ Checkpoint not found: {cfg['checkpoint']}")
                continue

            model.to(DEVICE).eval()
            models[name] = model
            print(f"      ✅ Successfully loaded")

        except Exception as e:
            print(f"      ❌ Failed: {e}")

    return models

# ==============================================================================
# EVALUATION LOOP
# ==============================================================================

def evaluate(models, model_configs, test_data):
    all_metrics = []

    print("\n" + "="*70)
    print("STARTING FLARE22 TEST SET EVALUATION")
    print("="*70)

    for sample in tqdm(test_data, desc="Evaluating"):
        gt = sample['gt']
        organ = sample['organ']
        spacing = sample['spacing']

        # Resize GT to 1024
        if gt.shape != (1024, 1024):
            gt_resized = cv2.resize(gt.astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        else:
            gt_resized = gt.astype(np.float32)

        for model_name, model in models.items():
            try:
                with torch.no_grad():
                    img_tensor, box_tensor = prepare_input(sample, model_configs[model_name]['type'])

                    if model_configs[model_name]['type'] == 'unet':
                        pred_logits = model(img_tensor)
                        pred = torch.sigmoid(pred_logits)
                    else:
                        pred = model(img_tensor, box_tensor)
                        if pred.min() < 0 or pred.max() > 1:
                            pred = torch.sigmoid(pred)

                    pred_np = pred.squeeze().cpu().numpy()
                    pred_binary = (pred_np > 0.5).astype(np.float32)

                    dice, nsd = compute_metrics(pred_binary, gt_resized, spacing)

                    all_metrics.append({
                        'Model': model_name,
                        'Organ': organ,
                        'Dice': dice,
                        'NSD': nsd
                    })

            except Exception as e:
                print(f"Error: {e}")
                all_metrics.append({'Model': model_name, 'Organ': organ, 'Dice': 0.0, 'NSD': 0.0})

    # Save Results
    df = pd.DataFrame(all_metrics)
    csv_path = os.path.join(SAVE_DIR, "flare22_detailed_results.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    overall = df.groupby('Model')[['Dice', 'NSD']].agg(['mean', 'std'])
    print("\nOverall Performance:\n", overall.round(2))

    organ_dice = df.pivot_table(values='Dice', index='Organ', columns='Model', aggfunc='mean')
    print("\nPer-Organ Dice:\n", organ_dice.round(2))

    # --- ADDED: Per-Organ NSD Table ---
    organ_nsd = df.pivot_table(values='NSD', index='Organ', columns='Model', aggfunc='mean')
    print("\nPer-Organ NSD:\n", organ_nsd.round(2))

    return df

if __name__ == "__main__":
    models = load_all_models(MODEL_CONFIGS, SAM_CKPT)
    test_data = load_test_data(TEST_DATA_DIR, ORGAN_MAPPING)
    if models and test_data:
        results = evaluate(models, MODEL_CONFIGS, test_data)
        print("\n✅ Evaluation complete!")