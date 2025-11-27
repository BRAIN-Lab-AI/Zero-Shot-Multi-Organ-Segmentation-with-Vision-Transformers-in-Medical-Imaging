# ============================================================================
# MedSAM++ FIXED - LoRA Bug Resolved
# ============================================================================

import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import monai
from tqdm import tqdm
import math
import glob
import cv2
from datetime import datetime
from scipy.ndimage import label as scipy_label, binary_erosion, binary_dilation
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes
from scipy.ndimage import distance_transform_edt, generate_binary_structure
import matplotlib.pyplot as plt

torch.manual_seed(2023)
np.random.seed(2023)
random.seed(2023)

# ==============================================================================
# ENHANCEMENT 1: IMPROVED AUTOMATIC PROMPT GENERATION (WITH ATLAS)
# ==============================================================================

class AtlasGuidedPromptGenerator:
    """Enhanced version with HU ranges and anatomical priors"""
    def __init__(self, modality='CT'):
        self.modality = modality
        
        # Atlas: Organ-specific characteristics (FLARE22 dataset)
        self.organ_atlas = {
            'liver': {
                'hu_range': (40, 180),
                'bbox_region': (0.30, 0.15, 0.75, 0.70),
                'min_area': 8000,
                'max_area': 120000
            },
            'right_kidney': {
                'hu_range': (20, 200),
                'bbox_region': (0.55, 0.30, 0.85, 0.70),
                'min_area': 1000,
                'max_area': 18000
            },
            'left_kidney': {
                'hu_range': (20, 200),
                'bbox_region': (0.15, 0.30, 0.45, 0.70),
                'min_area': 1000,
                'max_area': 18000
            },
            'spleen': {
                'hu_range': (40, 150),
                'bbox_region': (0.05, 0.20, 0.35, 0.60),
                'min_area': 1500,
                'max_area': 20000
            },
            'pancreas': {
                'hu_range': (30, 150),
                'bbox_region': (0.30, 0.35, 0.65, 0.60),
                'min_area': 500,
                'max_area': 8000
            }
        }
    
    def _apply_hu_threshold(self, image, hu_range):
        return ((image >= hu_range[0]) & (image <= hu_range[1])).astype(np.uint8) * 255
    
    def _filter_by_atlas_region(self, contours, atlas_region, img_shape):
        H, W = img_shape[:2]
        x_min, y_min, x_max, y_max = [int(r * d) for r, d in 
                                       zip(atlas_region, [W, H, W, H])]
        
        filtered = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            if x_min <= cx <= x_max and y_min <= cy <= y_max:
                filtered.append(contour)
        
        return filtered
    
    def generate_prompts(self, image_2d, target_organ=None):
        prompts = []
        
        if target_organ and target_organ in self.organ_atlas:
            atlas = self.organ_atlas[target_organ]
            
            if self.modality == 'CT':
                binary = self._apply_hu_threshold(image_2d, atlas['hu_range'])
            else:
                low, high = np.percentile(image_2d, [30, 80])
                binary = ((image_2d >= low) & (image_2d <= high)).astype(np.uint8) * 255
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = self._filter_by_atlas_region(contours, atlas['bbox_region'], image_2d.shape)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if atlas['min_area'] <= area <= atlas['max_area']:
                    x, y, w, h = cv2.boundingRect(contour)
                    pad = 10
                    prompts.append([
                        max(0, x - pad),
                        max(0, y - pad),
                        min(image_2d.shape[1], x + w + pad),
                        min(image_2d.shape[0], y + h + pad)
                    ])
        else:
            img_uint8 = (image_2d * 255 / image_2d.max()).astype(np.uint8) if image_2d.max() > 255 else image_2d.astype(np.uint8)
            
            for thresh in [50, 100, 150]:
                _, binary = cv2.threshold(img_uint8, thresh, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 300 < area < 80000:
                        x, y, w, h = cv2.boundingRect(contour)
                        if 0.2 < w/h < 5:
                            prompts.append([max(0, x-10), max(0, y-10),
                                          min(img_uint8.shape[1], x+w+10),
                                          min(img_uint8.shape[0], y+h+10)])
        
        return prompts[:5] if prompts else None


# ==============================================================================
# ENHANCEMENT 2: LEARNABLE 2.5D FUSION
# ==============================================================================

class SliceFusionModule(nn.Module):
    """Learnable multi-slice fusion"""
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


class MedSAMDataset25D(Dataset):
    """Enhanced dataset with 5-slice context"""
    def __init__(self, data_root, num_slices=5, bbox_shift=20, use_auto_prompt=True, target_organ=None):
        self.data_root = data_root
        self.num_slices = num_slices
        self.pad = num_slices // 2
        self.bbox_shift = bbox_shift
        self.use_auto_prompt = use_auto_prompt
        self.target_organ = target_organ
        
        self.gt_path = os.path.join(data_root, "gts")
        self.img_path = os.path.join(data_root, "imgs")
        self.gt_files = sorted(glob.glob(os.path.join(self.gt_path, "*.npy")))
        
        self.prompt_gen = AtlasGuidedPromptGenerator(modality='CT')
        
        print(f"Dataset: {len(self.gt_files)} samples, {num_slices}-slice context")
    
    def __len__(self):
        return len(self.gt_files)
    
    def _load_slice(self, path):
        if os.path.exists(path):
            try:
                img = np.load(path, allow_pickle=True)
                if len(img.shape) == 3:
                    return img[:, :, 0]
                return img
            except Exception as e:
                return np.zeros((1024, 1024), dtype=np.float32)
        return np.zeros((1024, 1024), dtype=np.float32)
    
    def _get_neighbor_path(self, current_path, offset):
        try:
            base = os.path.basename(current_path)
            parts = base.rsplit('-', 1)
            if len(parts) == 2:
                name_id = parts[0]
                curr_idx = int(parts[1].split('.')[0])
                target_idx = curr_idx + offset
                neighbor = f"{name_id}-{target_idx:03d}.npy"
                full_path = os.path.join(self.img_path, neighbor)
                if os.path.exists(full_path):
                    return full_path
        except:
            pass
        return current_path.replace("gts", "imgs")
    
    def __getitem__(self, index):
        gt_path = self.gt_files[index]
        
        # Load multi-slice stack
        slices = []
        for offset in range(-self.pad, self.pad + 1):
            neighbor_path = self._get_neighbor_path(gt_path, offset)
            slice_img = self._load_slice(neighbor_path)
            if slice_img.max() > 0:
                slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min())
            slices.append(slice_img)
        
        multi_slice = np.stack(slices, axis=0)
        
        # Load GT
        gt = np.load(gt_path, allow_pickle=True)
        if gt.shape != (1024, 1024):
            gt = cv2.resize(gt, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        
        if np.sum(gt) > 0:
            label_ids = np.unique(gt)[1:]
            gt2D = np.uint8(gt == random.choice(label_ids.tolist()))
        else:
            gt2D = np.zeros_like(gt, dtype=np.uint8)
        
        # Simple augmentation
        if random.random() > 0.5:
            multi_slice = np.flip(multi_slice, axis=2).copy()
            gt2D = np.flip(gt2D, axis=1).copy()
        
        # Generate bbox
        bbox = None
        if self.use_auto_prompt and random.random() > 0.3:
            center_slice = multi_slice[self.pad]
            prompts = self.prompt_gen.generate_prompts(center_slice, self.target_organ)
            
            if prompts and gt2D.sum() > 0:
                y_t, x_t = np.where(gt2D > 0)
                gt_box = [np.min(x_t), np.min(y_t), np.max(x_t), np.max(y_t)]
                
                best_iou, best_prompt = 0, None
                for p in prompts:
                    xA = max(p[0], gt_box[0])
                    yA = max(p[1], gt_box[1])
                    xB = min(p[2], gt_box[2])
                    yB = min(p[3], gt_box[3])
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    if inter > best_iou:
                        best_iou = inter
                        best_prompt = p
                
                if best_prompt:
                    bbox = np.array(best_prompt)
        
        # Fallback to GT bbox
        if bbox is None:
            y_i, x_i = np.where(gt2D > 0)
            if len(y_i) > 0:
                pad = random.randint(0, self.bbox_shift)
                bbox = np.array([
                    max(0, np.min(x_i) - pad),
                    max(0, np.min(y_i) - pad),
                    min(1024, np.max(x_i) + pad),
                    min(1024, np.max(y_i) + pad)
                ])
            else:
                bbox = np.array([0, 0, 1024, 1024])
        
        return (
            torch.tensor(multi_slice).float(),
            torch.tensor(gt2D[None]).long(),
            torch.tensor(bbox).float()
        )


# ==============================================================================
# FIXED LORA IMPLEMENTATION
# ==============================================================================

class LoRALayer(nn.Module):
    """Fixed LoRA layer - properly registered parameters"""
    def __init__(self, original_linear, rank=4, alpha=16):
        super().__init__()
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # Properly register original layer as a module
        self.original = original_linear
        
        # Register LoRA parameters properly
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Original output
        result = self.original(x)
        # Add LoRA adaptation
        result = result + (x @ self.lora_A @ self.lora_B) * self.scaling
        return result


# ==============================================================================
# MODEL WITH FIXED LORA
# ==============================================================================

class MedSAM25D(nn.Module):
    """
    ✅ CORRECTED: Inject LoRA BEFORE freezing
    """
    def __init__(self, sam_model, num_slices=5, use_lora=True, lora_rank=4, freeze_encoder=True):
        super().__init__()
        self.image_encoder = sam_model.image_encoder
        self.mask_decoder = sam_model.mask_decoder
        self.prompt_encoder = sam_model.prompt_encoder
        
        # Step 1: Add fusion module
        self.slice_fusion = SliceFusionModule(num_slices)
        
        # Step 2: Inject LoRA (if using)
        if use_lora:
            self._inject_lora(lora_rank)
            print(f"✅ LoRA injected with rank={lora_rank}")
        
        # Step 3: Freeze parameters (AFTER LoRA injection)
        if freeze_encoder:
            frozen_count = 0
            for name, param in self.image_encoder.named_parameters():
                # Don't freeze LoRA parameters!
                if 'lora' not in name.lower():
                    param.requires_grad = False
                    frozen_count += 1
            print(f"✅ Froze {frozen_count} encoder parameters (LoRA kept trainable)")
        else:
            print(f"✅ Encoder kept trainable (unfrozen mode)")
        
        # Always freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        
        # Verify trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora_params = sum(p.numel() for n, p in self.named_parameters() if 'lora' in n.lower() and p.requires_grad)
        
        print(f"📊 Parameters: Total={total_params:,}, Trainable={trainable_params:,}, LoRA={lora_params:,}")
    
    def _inject_lora(self, rank):
        """Inject LoRA into attention QKV layers"""
        for name, module in self.image_encoder.named_modules():
            if "qkv" in name and isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = dict(self.image_encoder.named_modules())[parent_name]
                lora_layer = LoRALayer(module, rank=rank, alpha=16)
                setattr(parent, child_name, lora_layer)
    
    def forward(self, multi_slice, box):
        fused = self.slice_fusion(multi_slice)
        emb = self.image_encoder(fused)
        
        with torch.no_grad():
            box_torch = box[:, None, :]
            sparse, dense = self.prompt_encoder(points=None, boxes=box_torch, masks=None)
        
        masks, _ = self.mask_decoder(
            image_embeddings=emb,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False
        )
        
        return F.interpolate(masks, size=(1024, 1024), mode='bilinear', align_corners=False)

# ==============================================================================
# LOSSES
# ==============================================================================

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.focal = monai.losses.FocalLoss(reduction='mean')
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        return 0.4 * self.dice(pred, target) + 0.3 * self.focal(pred, target.float()) + 0.3 * self.bce(pred, target.float())


# ==============================================================================
# MAIN TRAINING
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--tr_npy_path", type=str, default="/content/fast_data/train")
    parser.add_argument("--val_npy_path", type=str, default="/content/fast_data/val")
    parser.add_argument("-checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth")
    parser.add_argument("-work_dir", type=str, default="./work_dir")
    parser.add_argument("-num_epochs", type=int, default=50)
    parser.add_argument("-batch_size", type=int, default=2)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-num_slices", type=int, default=5)
    parser.add_argument("--target_organ", type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device('cuda')
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Load model
    sam = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    model = MedSAM25D(sam, num_slices=args.num_slices, use_lora=True).to(device)
    
    # Optimizer (only trainable params)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    criterion = CombinedLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"🎯 Total trainable parameters: {sum(p.numel() for p in trainable):,}")
    
    # Verify LoRA is trainable
    lora_trainable = sum(p.numel() for n, p in model.named_parameters() if 'lora' in n.lower() and p.requires_grad)
    print(f"🎯 Trainable LoRA parameters: {lora_trainable:,}")
          
    # Datasets
    train_dataset = MedSAMDataset25D(args.tr_npy_path, num_slices=args.num_slices, 
                                      use_auto_prompt=True, target_organ=args.target_organ)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    val_loader = None
    if os.path.exists(args.val_npy_path):
        val_dataset = MedSAMDataset25D(args.val_npy_path, num_slices=args.num_slices, use_auto_prompt=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"🚀 Training MedSAM++ with {args.num_slices}-slice context + LoRA")
    
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.num_epochs):
        # Train
        model.train()
        train_loss = 0
        
        for img, gt, box in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            img, gt, box = img.to(device), gt.to(device), box.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                pred = model(img, box)
                gt_resized = F.interpolate(gt.float(), size=pred.shape[-2:], mode='nearest')
                loss = criterion(pred, gt_resized)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step()
        train_losses.append(train_loss)
        
        # Validate
        if val_loader:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for img, gt, box in val_loader:
                    img, gt, box = img.to(device), gt.to(device), box.to(device)
                    pred = model(img, box)
                    gt_resized = F.interpolate(gt.float(), size=pred.shape[-2:], mode='nearest')
                    val_loss += criterion(pred, gt_resized).item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = os.path.join(args.work_dir, "MedSAM++_with_LoRA_best-unfrozen.pth")
                torch.save(model.state_dict(), save_path)
                print(f"  ✅ Best model saved! (LoRA params included)")
        else:
            val_losses.append(None)
            print(f"Epoch {epoch+1}: Train={train_loss:.4f}")
            torch.save(model.state_dict(), os.path.join(args.work_dir, "unfrozen-MedSAM++_with_LoRA_latest.pth"))
    
    # Verify saved checkpoint has LoRA
    saved_state = torch.load(os.path.join(args.work_dir, "MedSAM++_with_LoRA_best-unfrozen.pth" if val_loader else "unfrozen-MedSAM++_with_LoRA_latest.pth"))
    lora_keys = [k for k in saved_state.keys() if 'lora' in k.lower()]
    print(f"\n✅ Checkpoint verification: {len(lora_keys)} LoRA keys saved")
    
    # Plot and save loss curves
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, args.num_epochs + 1)
    plt.plot(epochs_range, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=4)
    
    if val_loader:
        plt.plot(epochs_range, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('MedSAM++ Training (with LoRA Fixed)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loss_curve_path = os.path.join(args.work_dir, 'unfrozen-medsam++_lora_fixed_loss_curve.png')
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Loss curve saved to: {loss_curve_path}")
    plt.close()
    
    print("\n✅ Training complete with LoRA properly saved!")


if __name__ == "__main__":
    main()