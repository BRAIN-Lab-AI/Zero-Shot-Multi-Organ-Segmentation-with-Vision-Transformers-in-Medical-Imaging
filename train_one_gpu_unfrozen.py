# =============================================================================
# STEP 3: Train Baseline (With Validation Loop)
# =============================================================================
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
import glob
from datetime import datetime
import matplotlib.pyplot as plt

# Set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = os.path.join(data_root, "gts")
        self.img_path = os.path.join(data_root, "imgs")
        self.gt_path_files = sorted(glob.glob(os.path.join(self.gt_path, "*.npy")))
        self.bbox_shift = bbox_shift

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        img = np.load(os.path.join(self.img_path, img_name), allow_pickle=True)
        img = np.transpose(img, (2, 0, 1))
        gt = np.load(self.gt_path_files[index], allow_pickle=True)
        
        if np.sum(gt) > 0:
            label_ids = np.unique(gt)[1:]
            gt = np.uint8(gt == random.choice(label_ids.tolist())) if len(label_ids) > 0 else np.zeros_like(gt, dtype=np.uint8)
        else:
            gt = np.zeros_like(gt, dtype=np.uint8)
        
        y_i, x_i = np.where(gt > 0)
        if len(y_i) > 0:
            x_min, x_max = np.min(x_i), np.max(x_i)
            y_min, y_max = np.min(y_i), np.max(y_i)
            H, W = gt.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
            bbox = np.array([x_min, y_min, x_max, y_max])
        else:
            bbox = np.array([0, 0, 256, 256])
        
        return torch.tensor(img).float(), torch.tensor(gt[None, :, :]).long(), torch.tensor(bbox).float()

class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        
        # Freeze prompt encoder (always frozen)
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        
        # ⭐ UNFREEZE LAST 2 BLOCKS OF IMAGE ENCODER
        # First freeze all
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        # Then unfreeze last 2 transformer blocks (blocks.10 and blocks.11)
        for name, param in self.image_encoder.named_parameters():
            if 'blocks.10' in name or 'blocks.11' in name:
                param.requires_grad = True
                print(f"✅ Unfrozen: {name}")

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            sparse, dense = self.prompt_encoder(points=None, boxes=box_torch, masks=None)
        
        low, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False)
        
        return F.interpolate(low, size=(image.shape[2], image.shape[3]), mode="bilinear", align_corners=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tr_npy_path", type=str, default="/content/fast_data/train")
    parser.add_argument("--val_npy_path", type=str, default="/content/fast_data/val")
    parser.add_argument("--task_name", type=str, default="MedSAM-ViT-B")
    parser.add_argument("--checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth")
    parser.add_argument("--work_dir", type=str, default="./work_dir")
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    # Google Drive save path
    drive_save_path = "/content/drive/MyDrive/Colab Notebooks/NN_Deep_Learning_Term-Project/MedSAM"
    os.makedirs(drive_save_path, exist_ok=True)
    
    device = torch.device("cuda:0")
    sam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    medsam_model = MedSAM(sam_model.image_encoder, sam_model.mask_decoder, sam_model.prompt_encoder).to(device)
    
    # Get all trainable parameters (decoder + unfrozen encoder blocks)
    trainable_params = [p for p in medsam_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    print(f"🎯 Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    
    # Datasets
    train_dataset = NpyDataset(args.tr_npy_path)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Validation Loader
    if os.path.exists(args.val_npy_path):
        val_dataset = NpyDataset(args.val_npy_path)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        print(f"🚀 Starting Baseline Training with Validation ({len(val_dataset)} val samples)...")
    else:
        val_dataloader = None
        print("⚠️ Validation folder not found. Running without validation.")
    
    best_val_loss = 1e10
    losses = []
    val_losses = []
    
    for epoch in range(args.num_epochs):
        # --- TRAINING ---
        medsam_model.train()
        epoch_loss = 0
        for step, (image, gt, boxes) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1} Train")):
            optimizer.zero_grad()
            image, gt, boxes = image.to(device), gt.to(device), boxes.to(device)
            gt = F.interpolate(gt.float(), size=(1024, 1024), mode="nearest")
            pred = medsam_model(image, boxes.detach().cpu().numpy())
            loss = seg_loss(pred, gt) + ce_loss(pred, gt.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_dataloader)
        losses.append(epoch_loss)
        
        # --- VALIDATION ---
        if val_dataloader:
            medsam_model.eval()
            val_loss = 0
            with torch.no_grad():
                for image, gt, boxes in val_dataloader:
                    image, gt, boxes = image.to(device), gt.to(device), boxes.to(device)
                    gt = F.interpolate(gt.float(), size=(1024, 1024), mode="nearest")
                    pred = medsam_model(image, boxes.detach().cpu().numpy())
                    loss = seg_loss(pred, gt) + ce_loss(pred, gt.float())
                    val_loss += loss.item()
            
            val_loss /= len(val_dataloader)
            val_losses.append(val_loss)
            print(f'Epoch {epoch+1}: Train Loss={epoch_loss:.4f} | Val Loss={val_loss:.4f}')
            
            # Save Best Model based on VALIDATION Loss to Google Drive
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_save_path = os.path.join(drive_save_path, "Unfrozen_2_50-60-NN_MedSAM-Baseline_best.pth")
                torch.save(medsam_model.state_dict(), model_save_path)
                print(f" --> Best Val Loss! Model Saved to {model_save_path}")
        else:
            # Fallback if no validation set (save by training loss)
            print(f'Epoch {epoch+1}: Train Loss={epoch_loss:.4f}')
            model_save_path = os.path.join(drive_save_path, "Unfrozen_2_50-60-NN_MedSAM-Baseline_best.pth")
            torch.save(medsam_model.state_dict(), model_save_path)
    
    # Plotting and saving to Google Drive
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Train Loss", marker='o')
    if val_losses:
        plt.plot(val_losses, label="Val Loss", marker='s')
    plt.title("Baseline Training Curve")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    
    # Save plot to Google Drive
    plot_save_path = os.path.join(drive_save_path, "unfrozen_2_baseline_loss_curve.png")
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Loss curve saved to {plot_save_path}")
    
    print(f"✅ Baseline Training Finished.")
    print(f"📁 All files saved to: {drive_save_path}")

if __name__ == "__main__":
    main()