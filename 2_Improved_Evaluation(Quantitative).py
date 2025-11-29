# ==============================================================================
# GRAND COMPARISON: IMPROVED ARCHITECTURES ONLY + FULL VALIDATION
# ==============================================================================

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
import nibabel as nib
from scipy import ndimage
import math

# Metrics & Base Architectures
from monai.metrics import compute_dice, compute_surface_dice
from segment_anything import sam_model_registry

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "/content/drive/MyDrive/Colab Notebooks/NN_Deep_Learning_Term-Project/MedSAM/Improved-in_sha_Allah-Grand_Evaluation_Final_Improved_Only"
os.makedirs(SAVE_DIR, exist_ok=True)

# Data Paths
TEST_DATA_DIR = "/content/drive/MyDrive/Colab Notebooks/NN_Deep_Learning_Term-Project/MedSAM/data/npy/CT_Abd/test"
AMOS_IMG_DIR = "/content/drive/MyDrive/Colab Notebooks/NN_Deep_Learning_Term-Project/amos_dataset/imagesVa"
AMOS_LBL_DIR = "/content/drive/MyDrive/Colab Notebooks/NN_Deep_Learning_Term-Project/amos_dataset/labelsVa"

SAM_CKPT = "work_dir/SAM/sam_vit_b_01ec64.pth"

# FLARE22 Spacing: (Z, Y, X) - Used for Internal Volume-Based Validation
# NOTE: Z is 2.5mm (thick slices), X/Y are ~0.64mm
FLARE22_SPACING_3D = (2.5, 0.644531, 0.644531)
NSD_TOLERANCE = 2.0 # mm

# AMOS Spacing: (X, Y) - Used for External Slice-Based Validation
AMOS_SPACING_2D = (0.61588544, 0.61588544)

# Model Definitions (Improved Only)
MODEL_CONFIGS = {
    "Improved-Frozen": {
        "class": "Improved_Frozen",
        "checkpoint": "/content/drive/MyDrive/Colab Notebooks/NN_Deep_Learning_Term-Project/MedSAM/work_dir/MedSAM++_with_LoRA_best-frozen.pth",
        "color": "#96CEB4",
        "input_type": "2.5d"
    },
    "Improved-Unfrozen": {
        "class": "Improved_Unfrozen",
        "checkpoint": "/content/drive/MyDrive/Colab Notebooks/NN_Deep_Learning_Term-Project/MedSAM/work_dir/MedSAM++_with_LoRA_best-unfrozen.pth",
        "color": "#FFEAA7",
        "input_type": "2.5d"
    }
}

# FLARE22 -> AMOS Mapping
ORGAN_MAPPING = {
    "Liver": (1, 6),
    "Right Kidney": (2, 2),
    "Spleen": (3, 1),
    "Pancreas": (4, 10),
    "Left Kidney": (13, 3)
}

# ==============================================================================
# 1. SHARED MODULES (LoRA, Fusion, Assembler)
# ==============================================================================

class SliceFusionModule(nn.Module):
    """Learnable multi-slice fusion (3D -> 2D)"""
    def __init__(self, num_slices=5):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(num_slices, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 3, kernel_size=1)
        )

    def forward(self, x):
        return self.fusion(x)

class LoRALayer(nn.Module):
    """Low-Rank Adaptation Layer"""
    def __init__(self, original_layer, rank=4, alpha=16):
        super().__init__()
        self.original = original_layer
        self.lora_A = nn.Parameter(torch.zeros(original_layer.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, original_layer.out_features))
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

def inject_lora(model, rank=4):
    for name, module in model.image_encoder.named_modules():
        if "qkv" in name and isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = dict(model.image_encoder.named_modules()).get(parent_name, model.image_encoder)
            setattr(parent, child_name, LoRALayer(module, rank=rank))
    return model

class VolumeAssembler3D:
    """Refines stacked 2D predictions into a 3D volume"""
    def __init__(self):
        self.structure = np.ones((3, 3, 3), dtype=np.int32)

    def refine(self, volume):
        # Keep largest connected component
        labeled, num_feats = ndimage.label(volume, structure=self.structure)
        if num_feats > 1:
            sizes = [np.sum(labeled == i) for i in range(1, num_feats + 1)]
            largest = np.argmax(sizes) + 1
            volume = (labeled == largest).astype(np.uint8)
        # Morphological closing
        volume = ndimage.binary_closing(volume, structure=self.structure, iterations=1)
        return volume.astype(np.uint8)

# ==============================================================================
# 2. MODEL ARCHITECTURES (IMPROVED ONLY)
# ==============================================================================

class Improved_Frozen(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder, num_slices=5):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.slice_fusion = SliceFusionModule(num_slices=num_slices)
        inject_lora(self, rank=4)

    def forward(self, multi_slice, box):
        fused = self.slice_fusion(multi_slice)
        image_embedding = self.image_encoder(fused)
        with torch.no_grad():
            box_torch = box[:, None, :]
            sparse, dense = self.prompt_encoder(points=None, boxes=box_torch, masks=None)
        low, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False)
        return F.interpolate(low, size=(1024, 1024), mode="bilinear", align_corners=False)

class Improved_Unfrozen(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder, num_slices=5):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.slice_fusion = SliceFusionModule(num_slices=num_slices)
        inject_lora(self, rank=4)

    def forward(self, multi_slice, box):
        fused = self.slice_fusion(multi_slice)
        image_embedding = self.image_encoder(fused)
        with torch.no_grad():
            box_torch = box[:, None, :]
            sparse, dense = self.prompt_encoder(points=None, boxes=box_torch, masks=None)
        low, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False)
        return F.interpolate(low, size=(1024, 1024), mode="bilinear", align_corners=False)

# ==============================================================================
# 3. HELPER FUNCTIONS (Corrected Metrics)
# ==============================================================================

def load_model(name, config):
    print(f"Loading {name}...")
    ckpt_path = config["checkpoint"]
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)

    if config["class"] == "Improved_Frozen":
        model = Improved_Frozen(sam.image_encoder, sam.mask_decoder, sam.prompt_encoder)
    elif config["class"] == "Improved_Unfrozen":
        model = Improved_Unfrozen(sam.image_encoder, sam.mask_decoder, sam.prompt_encoder)
    else:
        raise ValueError(f"Unknown class: {config['class']}")

    model.to(DEVICE)

    if os.path.exists(ckpt_path):
        try:
            # FIX: Weights only=False for PyTorch 2.6+ compat
            state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            if "model" in state_dict: state_dict = state_dict["model"]
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded weights for {name}")
        except Exception as e:
            print(f"⚠️ Error loading weights for {name}: {e}")
    else:
        print(f"❌ Checkpoint not found: {ckpt_path}")

    model.eval()
    return model

def get_bbox(mask):
    y, x = np.where(mask > 0)
    if len(y) == 0: return np.array([0, 0, 1024, 1024])
    return np.array([np.min(x), np.min(y), np.max(x), np.max(y)])

def compute_metrics(pred, gt, spacing=None):
    """
    Computes Dice and NSD (Normalized Surface Dice).
    Returns metrics as PERCENTAGES (0-100).
    """
    # 1. Compute Dice (0-1)
    dice_raw = compute_dice(pred, gt).item()
    # Handle potential NaNs if prediction is empty
    if math.isnan(dice_raw): dice_raw = 0.0

    # 2. Compute NSD (0-1)
    try:
        if spacing:
            nsd_raw = compute_surface_dice(pred, gt, class_thresholds=[NSD_TOLERANCE], spacing=spacing).item()
        else:
            nsd_raw = compute_surface_dice(pred, gt, class_thresholds=[NSD_TOLERANCE]).item()

        if math.isnan(nsd_raw): nsd_raw = 0.0
    except Exception:
        nsd_raw = 0.0

    # 3. CONVERT TO PERCENTAGES (0-100)
    return dice_raw * 100.0, nsd_raw * 100.0

def prepare_25d_input(img_vol, index, num_slices=5):
    """Extract 5-slice stack from volume (H, W, D) or list of slices"""
    pad = num_slices // 2
    stack = []
    is_3d_vol = isinstance(img_vol, np.ndarray) and img_vol.ndim == 3
    vol_len = img_vol.shape[2] if is_3d_vol else len(img_vol)

    for i in range(-pad, pad + 1):
        idx = min(max(index + i, 0), vol_len - 1)
        if is_3d_vol:
            slice_img = img_vol[:, :, idx]
        else:
            slice_img = img_vol[idx]
            if len(slice_img.shape) == 3: slice_img = slice_img[:, :, 0]
        if slice_img.max() > 0:
            slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min())
        stack.append(slice_img)
    return np.stack(stack)

# ==============================================================================
# 4. PLOTTING FUNCTIONS
# ==============================================================================

def create_error_map(pred, gt):
    error_map = np.zeros((*pred.shape, 3), dtype=np.uint8)
    tp = np.logical_and(pred == 1, gt == 1)
    fp = np.logical_and(pred == 1, gt == 0)
    fn = np.logical_and(pred == 0, gt == 1)
    error_map[tp] = [0, 255, 0] # Green = Correct
    error_map[fp] = [255, 0, 0] # Red = FP
    error_map[fn] = [0, 0, 255] # Blue = FN
    return error_map

def visualize_comparison(image, gt, predictions, model_names, save_path):
    n_models = len(predictions)
    fig, axes = plt.subplots(2, n_models + 2, figsize=(4 * (n_models + 2), 8))

    # Input & GT
    axes[0, 0].imshow(image, cmap='gray'); axes[0, 0].set_title("Input"); axes[0,0].axis('off')
    axes[0, 1].imshow(image, cmap='gray')
    axes[0, 1].imshow(gt, cmap='Greens', alpha=0.5); axes[0, 1].set_title("GT"); axes[0,1].axis('off')

    # Predictions
    for idx, (pred, name) in enumerate(zip(predictions, model_names)):
        axes[0, idx + 2].imshow(image, cmap='gray')
        axes[0, idx + 2].imshow(pred, cmap='Reds', alpha=0.5)
        axes[0, idx + 2].set_title(name); axes[0, idx + 2].axis('off')

    # Error Maps
    axes[1, 0].axis('off'); axes[1, 1].axis('off')
    for idx, (pred, name) in enumerate(zip(predictions, model_names)):
        err = create_error_map(pred, gt)
        axes[1, idx + 2].imshow(err); axes[1, idx + 2].set_title("Error Map"); axes[1, idx + 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_organ_wise_comparison(results_df, save_dir, suffix=""):
    """Updated to plot both Dice and NSD"""
    # 1. Plot Dice
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=results_df, x='Organ', y='Dice', hue='Model', ax=ax)
    ax.set_title(f'Dice Score Distribution by Organ {suffix}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/organ_wise_dice_box{suffix}.png", dpi=300)
    plt.close()

    # 2. Plot NSD (ADDITION)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=results_df, x='Organ', y='NSD', hue='Model', ax=ax)
    ax.set_title(f'NSD Score Distribution by Organ {suffix}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/organ_wise_nsd_box{suffix}.png", dpi=300)
    plt.close()

# ==============================================================================
# 5. INTERNAL VALIDATION (Volume-Based with Spacing)
# ==============================================================================

def internal_validation(models, save_dir):
    print("\n🔬 STARTING INTERNAL VALIDATION (Volume-Based)")
    print(f"ℹ️  Using Spacing: {FLARE22_SPACING_3D} (D, H, W)")

    img_files = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "imgs", "*.npy")))

    cases = {}
    for f in img_files:
        case_id = os.path.basename(f).split('-')[0]
        cases.setdefault(case_id, []).append(f)

    all_results = []
    assembler = VolumeAssembler3D()

    for case_id, files in tqdm(cases.items(), desc="Processing Internal Cases"):
        files.sort()
        vol_imgs = [np.load(f) for f in files]
        vol_gts = [np.load(f.replace("imgs", "gts")) for f in files]

        vol_imgs = [cv2.resize(img, (1024, 1024)) if img.shape[:2] != (1024, 1024) else img for img in vol_imgs]
        vol_gts = [cv2.resize(gt, (1024, 1024), interpolation=cv2.INTER_NEAREST) if gt.shape != (1024, 1024) else gt for gt in vol_gts]

        model_volumes = {name: [] for name in models.keys()}

        for idx, (img, gt) in enumerate(zip(vol_imgs, vol_gts)):
            stack_25d = prepare_25d_input(vol_imgs, idx, num_slices=5)
            img_25d_t = torch.tensor(stack_25d).float().unsqueeze(0).to(DEVICE)

            gt_256 = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_NEAREST)
            bbox = get_bbox(gt_256)
            bbox_t = torch.tensor([bbox * 4]).float().to(DEVICE)

            for name, model in models.items():
                with torch.no_grad():
                    pred = (torch.sigmoid(model(img_25d_t, bbox_t)) > 0.5).float()
                model_volumes[name].append(pred.squeeze().cpu().numpy().astype(np.uint8))

        gt_vol_t = torch.tensor(np.stack(vol_gts)).unsqueeze(0).unsqueeze(0).to(DEVICE)

        for name in models.keys():
            raw_vol = np.stack(model_volumes[name])
            refined_vol = assembler.refine(raw_vol)
            pred_t = torch.tensor(refined_vol).unsqueeze(0).unsqueeze(0).to(DEVICE)

            dice, nsd = compute_metrics(pred_t, gt_vol_t, spacing=FLARE22_SPACING_3D)

            unique_lbls = np.unique(np.stack(vol_gts))
            organ_id = unique_lbls[1] if len(unique_lbls) > 1 else 0
            organ_name = f"Organ_{organ_id}"

            all_results.append({
                "Case": case_id, "Model": name, "Organ": organ_name,
                "Dice": dice, "NSD": nsd
            })

        mid_idx = len(vol_imgs) // 2
        vis_preds = [model_volumes[name][mid_idx] for name in models.keys()]
        visualize_comparison(vol_imgs[mid_idx], vol_gts[mid_idx], vis_preds, list(models.keys()),
                             f"{save_dir}/internal_{case_id}.png")

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(save_dir, "internal_results.csv"), index=False)
    plot_organ_wise_comparison(df, save_dir, suffix="_internal")
    return df

# ==============================================================================
# 6. EXTERNAL VALIDATION (AMOS Dataset)
# ==============================================================================

def external_validation_amos(models, save_dir):
    print("\n🌍 STARTING EXTERNAL VALIDATION (AMOS)")
    print(f"ℹ️  Using Spacing (In-Plane): {AMOS_SPACING_2D}")

    img_files = sorted(glob.glob(os.path.join(AMOS_IMG_DIR, "*.nii.gz")))
    lbl_files = sorted(glob.glob(os.path.join(AMOS_LBL_DIR, "*.nii.gz")))

    if not img_files:
        print("❌ AMOS dataset not found!")
        return None

    all_results = []

    for img_path, lbl_path in tqdm(zip(img_files[:5], lbl_files[:5]), total=min(5, len(img_files)), desc="AMOS Cases"):
        case_id = os.path.basename(img_path).split('.')[0]
        nii_img = nib.load(img_path).get_fdata()
        nii_lbl = nib.load(lbl_path).get_fdata()

        img_norm = np.clip(nii_img, -160, 240)
        img_norm = (img_norm - (-160)) / (240 - (-160)) * 255.0

        for organ_name, (flare_id, amos_id) in ORGAN_MAPPING.items():
            if np.sum(nii_lbl == amos_id) == 0: continue
            z_indices = np.unique(np.where(nii_lbl == amos_id)[2])

            for z in z_indices[::3]:
                img_slice = img_norm[:, :, z]
                gt_slice = (nii_lbl[:, :, z] == amos_id).astype(np.uint8)

                img_1024 = cv2.resize(img_slice, (1024, 1024))
                gt_1024 = cv2.resize(gt_slice, (1024, 1024), interpolation=cv2.INTER_NEAREST)

                stack_25d = prepare_25d_input(img_norm, z, num_slices=5)
                stack_resized = [cv2.resize(s, (1024, 1024)) for s in stack_25d]
                stack_25d = np.stack(stack_resized)
                img_25d_t = torch.tensor(stack_25d).float().unsqueeze(0).to(DEVICE)

                gt_256 = cv2.resize(gt_slice, (256, 256), interpolation=cv2.INTER_NEAREST)
                bbox = get_bbox(gt_256)
                bbox_t = torch.tensor([bbox * 4]).float().to(DEVICE)
                gt_t = torch.tensor(gt_1024).unsqueeze(0).unsqueeze(0).to(DEVICE)

                for name, model in models.items():
                    with torch.no_grad():
                        pred = (torch.sigmoid(model(img_25d_t, bbox_t)) > 0.5).float()

                        # Compute Percentage Metrics
                        dice, nsd = compute_metrics(pred, gt_t, spacing=AMOS_SPACING_2D)

                        all_results.append({
                            "Case": case_id, "Organ": organ_name, "Model": name,
                            "Dice": dice, "NSD": nsd
                        })

    if not all_results: return None
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(save_dir, "external_amos_results.csv"), index=False)
    plot_organ_wise_comparison(df, save_dir, suffix="_external")
    return df

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("="*60)
    print("🚀 GRAND COMPARISON: IMPROVED ARCHITECTURES + FULL VALIDATION")
    print("="*60)

    models = {}
    for name, config in MODEL_CONFIGS.items():
        models[name] = load_model(name, config)

    df_int = internal_validation(models, SAVE_DIR)
    df_ext = external_validation_amos(models, SAVE_DIR)

    print("\n" + "="*60)
    print("📊 INTERNAL SUMMARY (Metrics: Mean % ± Std)")
    print("="*60)
    if df_int is not None:
        # Aggregates mean and std for both metrics
        summary_int = df_int.groupby("Model")[["Dice", "NSD"]].agg(["mean", "std"])
        print(summary_int.round(2))

        # --- NEW: Organ-wise Performance Table (Internal) ---
        print("\n🔎 Organ-wise Performance (Internal):")
        organ_stats_int = df_int.groupby(['Model', 'Organ'])[['Dice', 'NSD']].agg(['mean', 'std'])
        print(organ_stats_int.round(2))

    print("\n" + "="*60)
    print("🌍 EXTERNAL SUMMARY (Metrics: Mean % ± Std)")
    print("="*60)
    if df_ext is not None:
        # Aggregates mean and std for both metrics
        summary_ext = df_ext.groupby("Model")[["Dice", "NSD"]].agg(["mean", "std"])
        print(summary_ext.round(2))

        # --- NEW: Organ-wise Performance Table (External) ---
        print("\n🔎 Organ-wise Performance (External):")
        organ_stats_ext = df_ext.groupby(['Model', 'Organ'])[['Dice', 'NSD']].agg(['mean', 'std'])
        print(organ_stats_ext.round(2))

    print(f"\n✅ All results saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()