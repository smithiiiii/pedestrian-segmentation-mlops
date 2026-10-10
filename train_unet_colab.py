





!pip install -q segmentation-models-pytorch albumentations


!wget -q https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip -O PennFudanPed.zip
!unzip -q -o PennFudanPed.zip
!ls PennFudanPed


import os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

SEED        = 42
IMG_SIZE    = 256            
BATCH_SIZE  = 8
EPOCHS      = 20
LR          = 3e-4
VAL_SPLIT   = 0.2
DATA_ROOT   = "PennFudanPed"
SAVE_PATH   = "best_model.pt"
MEAN        = (0.485, 0.456, 0.406)   
STD         = (0.229, 0.224, 0.225)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_everything(SEED)
print("Device:", DEVICE)


class PennFudanDataset(Dataset):
    def __init__(self, root, files, transform=None):
        self.img_dir  = os.path.join(root, "PNGImages")
        self.mask_dir = os.path.join(root, "PedMasks")
        self.files = files            
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        image = np.array(Image.open(os.path.join(self.img_dir, name)).convert("RGB"))
        mask_name = name.replace(".png", "_mask.png")
        mask = np.array(Image.open(os.path.join(self.mask_dir, mask_name)))
        
        mask = (mask > 0).astype(np.uint8)

        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        mask = mask.float().unsqueeze(0)   
        return image, mask

train_tfms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])
val_tfms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

all_files = sorted(os.listdir(os.path.join(DATA_ROOT, "PNGImages")))
all_files = [f for f in all_files if f.endswith(".png")]
rng = random.Random(SEED)
rng.shuffle(all_files)
n_val = int(len(all_files) * VAL_SPLIT)
val_files, train_files = all_files[:n_val], all_files[n_val:]
print(f"Train: {len(train_files)} | Val: {len(val_files)}")

train_ds = PennFudanDataset(DATA_ROOT, train_files, train_tfms)
val_ds   = PennFudanDataset(DATA_ROOT, val_files,   val_tfms)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)  
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)


x, y = next(iter(train_loader))
print("Image batch:", x.shape, x.dtype, "| Mask batch:", y.shape, y.dtype,
      "| mask unique:", torch.unique(y).tolist())


model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
).to(DEVICE)

bce  = nn.BCEWithLogitsLoss()
dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

def criterion(logits, targets):
    return bce(logits, targets) + dice(logits, targets)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
use_amp = DEVICE == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, inter, union = 0.0, 0.0, 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        logits = model(imgs)
        total_loss += criterion(logits, masks).item() * imgs.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        inter += (preds * masks).sum().item()
        union += ((preds + masks) > 0).float().sum().item()
    n = len(loader.dataset)
    return total_loss / n, inter / (union + 1e-7)   

best_val_loss = float("inf")
history = {"train": [], "val": [], "iou": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits.float(), masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * imgs.size(0)
    scheduler.step()

    train_loss = running / len(train_loader.dataset)
    val_loss, val_iou = evaluate(val_loader)
    history["train"].append(train_loss); history["val"].append(val_loss); history["iou"].append(val_iou)

    saved = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state, SAVE_PATH)
        saved = "  <-- saved best_model.pt"

    print(f"Epoch {epoch:02d}/{EPOCHS} | train {train_loss:.4f} | val {val_loss:.4f} "
          f"| val IoU {val_iou:.4f} | lr {scheduler.get_last_lr()[0]:.2e}{saved}")

print(f"\nBest val loss: {best_val_loss:.4f}")


plt.figure(figsize=(6, 3.5))
plt.plot(history["train"], label="train loss")
plt.plot(history["val"], label="val loss")
plt.xlabel("epoch"); plt.ylabel("BCE + Dice"); plt.legend(); plt.grid(alpha=0.3)
plt.show()


check_model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
check_model.load_state_dict(torch.load(SAVE_PATH, map_location="cpu"))
check_model.to(DEVICE).eval()

def denorm(t):
    m = torch.tensor(MEAN).view(3, 1, 1); s = torch.tensor(STD).view(3, 1, 1)
    return (t.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()

n_show = min(5, len(val_ds))
fig, axes = plt.subplots(n_show, 3, figsize=(10, 3.2 * n_show))
with torch.no_grad():
    for i in range(n_show):
        img, gt = val_ds[i]
        probs = torch.sigmoid(check_model(img.unsqueeze(0).to(DEVICE)))[0, 0].cpu()
        pred = (probs > 0.5).numpy()
        print(f"Sample {i}: max prob = {probs.max():.3f} | predicted fg = {pred.mean()*100:.1f}% "
              f"| GT fg = {gt.mean().item()*100:.1f}%")
        axes[i, 0].imshow(denorm(img));          axes[i, 0].set_title("Original Image")
        axes[i, 1].imshow(gt[0], cmap="gray");   axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 2].imshow(pred, cmap="gray");    axes[i, 2].set_title("Predicted Mask (>0.5)")
        for a in axes[i]: a.axis("off")
plt.tight_layout(); plt.show()


from google.colab import files
files.download(SAVE_PATH)
