import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from tqdm import tqdm
from segment_anything import sam_model_registry
from monai.networks.nets import BasicUNet

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PATHS
TEST_DATA_DIR = "/content/drive/MyDrive/Colab Notebooks/NN_Deep_Learning_Term-Project/MedSAM/data/npy/CT_Abd/test"
SAM_CKPT = "work_dir/SAM/sam_vit_b_01ec64.pth"
SAVE_DIR = "/content/drive/MyDrive/Colab Notebooks/NN_Deep_Learning_Term-Project/MedSAM/Qualitative_Results_Advanced"

# Sub-directories for organization
VIZ_DIR = os.path.join(SAVE_DIR, "2D_Slices_and_Errors")
RENDER_DIR = os.path.join(SAVE_DIR, "3D_Renders")
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(RENDER_DIR, exist_ok=True)

# SPACING FOR 3D RENDERING (X, Y, Z)
FLARE22_SPACING = (0.644531, 0.644531, 2.5)

MODEL_CONFIGS = {
    "U-Net": {
        "checkpoint": "/content/drive/MyDrive/Colab Notebooks/NN_Deep_Learning_Term-Project/MedSAM/work_dir/50-60-NN_UNet_best_fixed.pth",
        "type": "unet", "input_mode": "2d"
    },
    "Baseline-Frozen": {
        "checkpoint": "/content/drive/MyDrive/Colab Notebooks/NN_Deep_Learning_Term-Project/MedSAM/work_dir/50-60-NN_MedSAM-Baseline_best.pth",
        "type": "baseline", "input_mode": "2d"
    },
    "Baseline-Unfrozen": {
        "checkpoint": "/content/drive/MyDrive/Colab Notebooks/NN_Deep_Learning_Term-Project/MedSAM/work_dir/Unfrozen_2_50-60-NN_MedSAM-Baseline_best.pth",
        "type": "baseline", "input_mode": "2d"
    },
    "Improved-Frozen": {
        "checkpoint": "/content/drive/MyDrive/Colab Notebooks/NN_Deep_Learning_Term-Project/MedSAM/work_dir/MedSAM++_with_LoRA_best-frozen.pth",
        "type": "improved_frozen", "input_mode": "2.5d"
    },
    "Improved-Unfrozen": {
        "checkpoint": "/content/drive/MyDrive/Colab Notebooks/NN_Deep_Learning_Term-Project/MedSAM/work_dir/MedSAM++_with_LoRA_best-unfrozen.pth",
        "type": "improved_unfrozen", "input_mode": "2.5d"
    }
}

ORGAN_MAPPING = {
    "Liver": 1,
    "Right Kidney": 2,
    "Spleen": 3,
    "Pancreas": 4,
    "Left Kidney": 3
}

# ==============================================================================
# 2. MODEL DEFINITIONS & HELPERS
# ==============================================================================

class SliceFusionModule(nn.Module):
    def __init__(self, num_slices=5):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(num_slices, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 3, kernel_size=1)
        )
    def forward(self, x): return self.fusion(x)

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=4, alpha=16):
        super().__init__()
        self.original = original_layer
        self.lora_A = nn.Parameter(torch.zeros(original_layer.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, original_layer.out_features))
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        for p in self.original.parameters(): p.requires_grad = False
    def forward(self, x): return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

def inject_lora(model, rank=4):
    for name, module in model.image_encoder.named_modules():
        if "qkv" in name and isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = dict(model.image_encoder.named_modules())[parent_name]
            setattr(parent, child_name, LoRALayer(module, rank=rank))
    return model

class MedSAM_Wrapper(nn.Module):
    def __init__(self, sam_model, use_fusion=False, num_slices=5):
        super().__init__()
        self.image_encoder = sam_model.image_encoder
        self.mask_decoder = sam_model.mask_decoder
        self.prompt_encoder = sam_model.prompt_encoder
        self.use_fusion = use_fusion
        if use_fusion:
            self.slice_fusion = SliceFusionModule(num_slices)

    def forward(self, img, box):
        if self.use_fusion: img = self.slice_fusion(img)
        else: img = self.image_encoder(img)
        if self.use_fusion: img = self.image_encoder(img)

        if box.dim() == 2: box = box.unsqueeze(1)
        with torch.no_grad():
            sparse, dense = self.prompt_encoder(points=None, boxes=box, masks=None)
        masks, _ = self.mask_decoder(
            image_embeddings=img,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False
        )
        return F.interpolate(masks, size=(1024, 1024), mode='bilinear', align_corners=False)

def get_bbox(mask):
    y, x = np.where(mask > 0)
    if len(y) == 0: return None
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    return np.array([x_min, y_min, x_max, y_max])

def load_models():
    models = {}
    print("Loading models...")
    for name, cfg in MODEL_CONFIGS.items():
        try:
            if cfg['type'] == 'unet':
                model = BasicUNet(spatial_dims=2, in_channels=3, out_channels=1, 
                                  features=(32, 32, 64, 128, 256, 32), dropout=0.1)
            else:
                sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)
                if 'improved' in cfg['type']:
                    inject_lora(sam, rank=4)
                    model = MedSAM_Wrapper(sam, use_fusion=True)
                else:
                    model = MedSAM_Wrapper(sam, use_fusion=False)
            
            if os.path.exists(cfg['checkpoint']):
                ckpt = torch.load(cfg['checkpoint'], map_location=DEVICE, weights_only=False)
                state_dict = ckpt.get('model', ckpt)
                model.load_state_dict(state_dict, strict=False)
                model.to(DEVICE).eval()
                models[name] = model
                print(f"✅ Loaded {name}")
            else:
                print(f"❌ Checkpoint missing: {name}")
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
    return models

def prepare_input(img, box, mode):
    if img.ndim == 3: img = img[:, :, 0]
    img = cv2.resize(img, (1024, 1024))
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    if mode == '2.5d':
        stack = np.stack([img]*5, axis=0) 
        t = torch.from_numpy(stack).float().unsqueeze(0).to(DEVICE)
    else:
        stack = np.stack([img]*3, axis=0)
        t = torch.from_numpy(stack).float().unsqueeze(0).to(DEVICE)
    
    box = box * (1024/img.shape[0]) 
    b_t = torch.from_numpy(box).float().unsqueeze(0).to(DEVICE)
    return t, b_t, img

# ==============================================================================
# 3. ERROR MAP GENERATION
# ==============================================================================

def create_error_map(pred_mask, gt_mask):
    """
    Creates an RGB Error Map:
    - Green: True Positive (Correct)
    - Red: False Positive (Leakage)
    - Blue: False Negative (Missed)
    """
    # Initialize blank RGB image
    error_map = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.float32)
    
    # 1. True Positives (Green) - Intersection
    tp = np.logical_and(pred_mask == 1, gt_mask == 1)
    error_map[tp] = [0, 1, 0] # RGB
    
    # 2. False Positives (Red) - Leakage
    fp = np.logical_and(pred_mask == 1, gt_mask == 0)
    error_map[fp] = [1, 0, 0]
    
    # 3. False Negatives (Blue) - Missed
    fn = np.logical_and(pred_mask == 0, gt_mask == 1)
    error_map[fn] = [0, 0, 1]
    
    return error_map

# ==============================================================================
# 4. 2D VISUALIZATION WITH ERROR MAPS
# ==============================================================================

def visualize_organ_with_errors(img, gt_mask, box, organ_name, sample_id, models):
    """
    Generates two rows: 
    Row 1: Segmentation overlays
    Row 2: Error Maps (Green=Correct, Red=Leakage, Blue=Missed)
    """
    n_models = len(models)
    fig, axes = plt.subplots(2, n_models + 2, figsize=(20, 8))
    
    # --- ROW 1: PREDICTIONS ---
    # 1. Input + Box
    axes[0, 0].imshow(img, cmap='gray')
    x1, y1, x2, y2 = box
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='blue', linewidth=2)
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title(f"Input: {organ_name}", fontsize=10)
    axes[0, 0].axis('off')

    # 2. GT
    axes[0, 1].imshow(img, cmap='gray')
    axes[0, 1].imshow(gt_mask, cmap='jet', alpha=0.5)
    axes[0, 1].set_title("Ground Truth", fontsize=10)
    axes[0, 1].axis('off')

    # Placeholder for error maps in first 2 cols of 2nd row
    axes[1, 0].axis('off')
    axes[1, 1].text(0.5, 0.5, "Error Map Legend:\n\nGreen: Correct\nRed: Leakage (FP)\nBlue: Missed (FN)", 
                    ha='center', va='center', fontsize=12)
    axes[1, 1].axis('off')

    # 3. Models
    for i, (name, model) in enumerate(models.items()):
        cfg = MODEL_CONFIGS[name]
        inp, b_t, _ = prepare_input(img, box, cfg['input_mode'])
        
        with torch.no_grad():
            if cfg['type'] == 'unet':
                pred = torch.sigmoid(model(inp))
            else:
                pred = torch.sigmoid(model(inp, b_t))
        
        pred_np = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        
        # Dice calculation for title
        dice = (2. * (pred_np * gt_mask).sum()) / (pred_np.sum() + gt_mask.sum() + 1e-8)
        
        # Plot Prediction (Row 1)
        axes[0, i+2].imshow(img, cmap='gray')
        axes[0, i+2].imshow(pred_np, cmap='Reds', alpha=0.5)
        axes[0, i+2].set_title(f"{name}\nDice: {dice:.2f}", fontsize=9)
        axes[0, i+2].axis('off')
        
        # Plot Error Map (Row 2)
        err_map = create_error_map(pred_np, gt_mask)
        axes[1, i+2].imshow(err_map)
        axes[1, i+2].set_title(f"Error Map: {name}", fontsize=9)
        axes[1, i+2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f"{organ_name}_{sample_id}.png"), dpi=150)
    plt.close()

# ==============================================================================
# 5. 3D SURFACE RENDERING
# ==============================================================================

def generate_3d_rendering(case_files, organ_name, organ_id, models):
    """
    Reconstructs a full 3D volume for a specific organ and renders the surface.
    """
    print(f"  Generating 3D Render for {organ_name}...")
    
    # 1. Load Volume & GT
    slices = []
    gts = []
    
    # Sort files to ensure correct Z-ordering
    case_files.sort()
    
    for f in case_files:
        img = np.load(f)
        if img.ndim == 3: img = img[:, :, 0]
        img = cv2.resize(img, (256, 256)) # Resize for memory efficiency in 3D
        slices.append(img)
        
        gt = np.load(f.replace("imgs", "gts"))
        gt_binary = (gt == organ_id).astype(np.uint8)
        gt_binary = cv2.resize(gt_binary, (256, 256), interpolation=cv2.INTER_NEAREST)
        gts.append(gt_binary)
        
    vol_np = np.stack(slices) # (D, H, W)
    gt_vol = np.stack(gts)
    
    if gt_vol.sum() == 0:
        print("    No organ found in this volume.")
        return

    # 2. Run Inference Slice-by-Slice for 3D reconstruction
    model_preds_3d = {name: [] for name in models.keys()}
    
    for i in range(len(slices)):
        # Get bounding box from GT (Oracle box for consistency in 3D)
        # In real scenario, you'd propagate the box, but for qualitative viz, this is standard
        bbox_slice = get_bbox(gts[i])
        if bbox_slice is None: 
            # If no organ on this slice, assume empty prediction (unless U-Net)
            for name in models: model_preds_3d[name].append(np.zeros((256, 256)))
            continue
            
        # Scale bbox back to 1024 for model, then resize output back to 256
        bbox_1024 = bbox_slice * 4 
        
        img_slice = cv2.resize(slices[i], (1024, 1024))
        img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
        
        for name, model in models.items():
            cfg = MODEL_CONFIGS[name]
            
            # Prepare input
            if cfg['input_mode'] == '2.5d':
                stack = np.stack([img_slice]*5, axis=0)
                t = torch.from_numpy(stack).float().unsqueeze(0).to(DEVICE)
            else:
                stack = np.stack([img_slice]*3, axis=0)
                t = torch.from_numpy(stack).float().unsqueeze(0).to(DEVICE)
            
            b_t = torch.from_numpy(bbox_1024).float().unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                if cfg['type'] == 'unet':
                    pred = torch.sigmoid(model(t))
                else:
                    pred = torch.sigmoid(model(t, b_t))
            
            pred_np = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            pred_small = cv2.resize(pred_np, (256, 256), interpolation=cv2.INTER_NEAREST)
            model_preds_3d[name].append(pred_small)

    # 3. Stack Predictions
    for name in model_preds_3d:
        model_preds_3d[name] = np.stack(model_preds_3d[name])

    # 4. Render using Marching Cubes
    fig = plt.figure(figsize=(15, 10))
    
    # Plot GT
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    try:
        verts, faces, _, _ = measure.marching_cubes(gt_vol, 0.5, spacing=FLARE22_SPACING)
        mesh = Poly3DCollection(verts[faces], alpha=0.70)
        face_color = [0.5, 0.5, 1]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        ax.set_xlim(0, gt_vol.shape[2]*FLARE22_SPACING[2]); ax.set_ylim(0, gt_vol.shape[1]*FLARE22_SPACING[1]); ax.set_zlim(0, gt_vol.shape[0]*FLARE22_SPACING[0])
        ax.set_title("Ground Truth")
    except: pass # Volume might be empty or too small

    # Plot Models
    for i, (name, vol) in enumerate(model_preds_3d.items()):
        ax = fig.add_subplot(2, 3, i+2, projection='3d')
        try:
            if vol.sum() > 100: # Only render if volume exists
                verts, faces, _, _ = measure.marching_cubes(vol, 0.5, spacing=FLARE22_SPACING)
                mesh = Poly3DCollection(verts[faces], alpha=0.70)
                mesh.set_facecolor([1, 0.5, 0.5]) # Reddish for models
                ax.add_collection3d(mesh)
                ax.set_xlim(0, vol.shape[2]*FLARE22_SPACING[2]); ax.set_ylim(0, vol.shape[1]*FLARE22_SPACING[1]); ax.set_zlim(0, vol.shape[0]*FLARE22_SPACING[0])
                ax.set_title(name)
            else:
                ax.text2D(0.5, 0.5, "No Volume", transform=ax.transAxes, ha="center")
        except: pass

    plt.tight_layout()
    plt.savefig(os.path.join(RENDER_DIR, f"{organ_name}_3D_Comparison.png"))
    plt.close()

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    models = load_models()
    
    img_files = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "imgs", "*.npy")))
    gt_dir = os.path.join(TEST_DATA_DIR, "gts")
    
    print("Generating Qualitative Results...")
    
    # 1. 2D Slices & Error Maps
    print("--- Generating 2D Slices & Error Maps ---")
    organs_done = {k: 0 for k in ORGAN_MAPPING.keys()}
    
    for img_path in tqdm(img_files):
        base = os.path.basename(img_path)
        gt_path = os.path.join(gt_dir, base)
        if not os.path.exists(gt_path): continue
        
        try:
            img = np.load(img_path)
            gt = np.load(gt_path)
            
            for organ, label_id in ORGAN_MAPPING.items():
                if organs_done[organ] >= 2: continue # 2 samples per organ
                
                if label_id in gt:
                    gt_binary = (gt == label_id).astype(np.uint8)
                    if gt_binary.sum() < 200: continue 
                    
                    gt_resized = cv2.resize(gt_binary, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                    bbox = get_bbox(gt_resized)
                    if bbox is None: continue
                    
                    visualize_organ_with_errors(img, gt_resized, bbox, organ, base, models)
                    organs_done[organ] += 1
                    
        except Exception as e: continue

    # 2. 3D Surface Rendering
    print("\n--- Generating 3D Surface Renders (One Case) ---")
    # Group files by Case ID (assuming format 'case_001-001.npy')
    case_groups = {}
    for f in img_files:
        case_id = os.path.basename(f).split('-')[0]
        if case_id not in case_groups: case_groups[case_id] = []
        case_groups[case_id].append(f)
    
    # Pick the first case that has enough slices
    target_case = None
    for cid, files in case_groups.items():
        if len(files) > 30: # Ensure reasonably volume size
            target_case = cid
            break
            
    if target_case:
        print(f"Rendering 3D volumes for Case: {target_case}")
        for organ, label_id in ORGAN_MAPPING.items():
            generate_3d_rendering(case_groups[target_case], organ, label_id, models)
    else:
        print("Not enough slices found for 3D rendering.")

    print(f"✅ Done! Check {SAVE_DIR} for all results.")