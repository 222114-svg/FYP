import json
import time
import logging
import numpy as np
import multiprocessing
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ── CONFIG ────────────────────────────────────────────────────────────────────
KP_DIR      = r"C:\Users\GPU\Downloads\asl_keypoints"
MODEL_DIR   = r"C:\Users\GPU\Downloads\asl_model"
CLASS_MAP   = r"C:\Users\GPU\Downloads\asl_videos\class_map.json"

SEQ_LEN      = 30
FEATURE_DIM  = 225
HIDDEN_SIZE  = 256   # smaller model = less overfitting on small data
NUM_LAYERS   = 2     # fewer layers = less overfitting
DROPOUT      = 0.6   # higher dropout = less overfitting
BATCH_SIZE   = 16    # smaller batch = more gradient updates per epoch
EPOCHS       = 150
LR           = 1e-3
WEIGHT_DECAY = 1e-3  # stronger regularization
PATIENCE     = 25    # more patience

# ── USE ALL DATA (train + val combined, then split 80/20) ─────────────────────
# With so little data, we can't afford to waste val samples
# We combine everything and do our own 80/20 split
COMBINE_SPLITS  = True   # combine train+val for more training data
VAL_RATIO       = 0.2    # 20% for validation

# ── CLASS SELECTION ───────────────────────────────────────────────────────────
TOP_N_CLASSES = 50    # reduce to 50 classes — more samples per class
MIN_SAMPLES   = 8     # only include classes with at least 8 samples
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            str(Path(MODEL_DIR) / "training.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────

class ASLDataset(Dataset):

    def __init__(self, samples, augment=False):
        """
        samples: list of (npy_path_str, label_int)
        """
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        seq = np.load(path).astype(np.float32)
        if self.augment:
            seq = self._augment(seq)
        return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)

    def _augment(self, seq):
        # Gaussian noise
        if np.random.rand() < 0.6:
            seq = seq + np.random.normal(0, 0.02, seq.shape).astype(np.float32)

        # Mirror hands
        if np.random.rand() < 0.5:
            lh = seq[:, :63].copy()
            rh = seq[:, 63:126].copy()
            seq[:, :63]    = rh
            seq[:, 63:126] = lh
            seq[:, 0:63:3]   = 1.0 - seq[:, 0:63:3]
            seq[:, 63:126:3] = 1.0 - seq[:, 63:126:3]

        # Temporal jitter
        if np.random.rand() < 0.5:
            drop_n   = np.random.randint(1, 6)
            keep_idx = sorted(np.random.choice(SEQ_LEN, SEQ_LEN - drop_n, replace=False))
            kept     = seq[keep_idx]
            x_old    = np.linspace(0, 1, len(keep_idx))
            x_new    = np.linspace(0, 1, SEQ_LEN)
            seq      = np.stack([
                np.interp(x_new, x_old, kept[:, i])
                for i in range(FEATURE_DIM)
            ], axis=1).astype(np.float32)

        # Random speed
        if np.random.rand() < 0.5:
            factor    = np.random.uniform(0.7, 1.3)
            new_len   = max(int(SEQ_LEN * factor), 2)
            x_old     = np.linspace(0, 1, SEQ_LEN)
            x_new     = np.linspace(0, 1, new_len)
            stretched = np.stack([
                np.interp(x_new, x_old, seq[:, i])
                for i in range(FEATURE_DIM)
            ], axis=1).astype(np.float32)
            if new_len >= SEQ_LEN:
                seq = stretched[:SEQ_LEN]
            else:
                pad = np.zeros((SEQ_LEN - new_len, FEATURE_DIM), dtype=np.float32)
                seq = np.vstack([stretched, pad])

        # Random scale
        if np.random.rand() < 0.4:
            scale = np.random.uniform(0.85, 1.15)
            seq   = seq * scale

        # Zero out random frames
        if np.random.rand() < 0.3:
            n_zero = np.random.randint(1, 4)
            zero_idx = np.random.choice(SEQ_LEN, n_zero, replace=False)
            seq[zero_idx] = 0.0

        return seq


def load_all_samples(kp_root, splits, label_remap):
    """Load samples from multiple splits into a flat list."""
    all_samples = []
    for split in splits:
        lj = Path(kp_root) / split / "labels.json"
        if not lj.exists():
            log.warning(f"  labels.json not found for split: {split} — skipping")
            continue
        records = json.loads(lj.read_text(encoding="utf-8"))
        count = 0
        for r in records:
            p     = Path(r["path"])
            label = int(r["label"])
            if not p.exists():
                continue
            if label not in label_remap:
                continue
            all_samples.append((str(p), label_remap[label]))
            count += 1
        log.info(f"  Loaded {count} samples from '{split}'")
    return all_samples


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL  (smaller to reduce overfitting)
# ─────────────────────────────────────────────────────────────────────────────

class AttentionPool(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.W = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        scores  = self.v(torch.tanh(self.W(x))).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * x).sum(dim=1)


class SignClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.lstm = nn.LSTM(
            input_size    = hidden_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )
        self.attn = AttentionPool(hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.attn(x)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def topk_accuracy(logits, labels, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        correct = pred.t().eq(labels.view(1, -1).expand_as(pred.t()))
        return {
            f"top{k}": correct[:k].any(dim=0).float().mean().item() * 100
            for k in topk
        }


def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = top1_sum = top5_sum = n = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for seqs, labels in loader:
            seqs   = seqs.to(DEVICE)
            labels = labels.to(DEVICE)

            if training:
                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    logits = model(seqs)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.amp.autocast("cuda"):
                    logits = model(seqs)
                    loss   = criterion(logits, labels)

            acc        = topk_accuracy(logits, labels)
            bs         = seqs.size(0)
            total_loss += loss.item() * bs
            top1_sum   += acc["top1"] * bs
            top5_sum   += acc["top5"] * bs
            n          += bs

    return total_loss / n, top1_sum / n, top5_sum / n


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("SignBridge — Step 3: Train Sign Classifier")
    log.info("=" * 60)
    log.info(f"Device      : {DEVICE}")
    if DEVICE.type == "cuda":
        log.info(f"GPU         : {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        log.warning("No GPU — training will be slow!")

    # ── read all training labels ──
    log.info("\nReading training labels...")
    train_lj = Path(KP_DIR) / "train" / "labels.json"
    if not train_lj.exists():
        log.error(f"Not found: {train_lj}  →  Run step2.py first!")
        exit(1)

    records = json.loads(train_lj.read_text(encoding="utf-8"))
    label_counts = Counter(
        int(r["label"])
        for r in records
        if Path(r["path"]).exists()
    )

    # also count val samples per label
    val_lj = Path(KP_DIR) / "val" / "labels.json"
    if val_lj.exists() and COMBINE_SPLITS:
        val_records = json.loads(val_lj.read_text(encoding="utf-8"))
        for r in val_records:
            if Path(r["path"]).exists():
                label_counts[int(r["label"])] += 1

    log.info(f"Total classes  : {len(label_counts)}")
    log.info(f"Total samples  : {sum(label_counts.values())}")

    # ── select TOP_N classes ──
    top_labels = [
        label for label, count in
        sorted(label_counts.items(), key=lambda x: -x[1])
        if count >= MIN_SAMPLES
    ][:TOP_N_CLASSES]

    total_selected = sum(label_counts[l] for l in top_labels)
    avg_per_class  = total_selected / len(top_labels)

    log.info(f"\nSelected top {len(top_labels)} classes")
    log.info(f"Total samples in selection : {total_selected}")
    log.info(f"Avg samples per class      : {avg_per_class:.1f}")
    log.info(f"Min samples per class      : {min(label_counts[l] for l in top_labels)}")
    log.info(f"Max samples per class      : {max(label_counts[l] for l in top_labels)}")

    if Path(CLASS_MAP).exists():
        cm        = json.loads(Path(CLASS_MAP).read_text(encoding="utf-8"))
        top_words = [cm.get(str(l), f"sign_{l}") for l in top_labels[:30]]
        log.info(f"Classes: {top_words}")

    label_remap = {orig: new for new, orig in enumerate(sorted(top_labels))}
    remap_inv   = {new: orig for orig, new in label_remap.items()}
    num_classes = len(label_remap)

    # ── load all samples ──
    log.info("\nLoading samples...")
    splits_to_load = ["train", "val"] if COMBINE_SPLITS else ["train"]
    all_samples = load_all_samples(KP_DIR, splits_to_load, label_remap)

    log.info(f"Total samples loaded : {len(all_samples)}")

    # shuffle and split 80/20
    np.random.shuffle(all_samples)
    n_val    = max(1, int(len(all_samples) * VAL_RATIO))
    n_train  = len(all_samples) - n_val
    train_samples = all_samples[:n_train]
    val_samples   = all_samples[n_train:]

    log.info(f"Train split : {len(train_samples)} samples")
    log.info(f"Val split   : {len(val_samples)} samples")

    train_ds = ASLDataset(train_samples, augment=True)
    val_ds   = ASLDataset(val_samples,   augment=False)

    num_workers = 6
    train_loader = DataLoader(
        train_ds,
        batch_size         = BATCH_SIZE,
        shuffle            = True,
        num_workers        = num_workers,
        pin_memory         = (DEVICE.type == "cuda"),
        persistent_workers = (num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size         = BATCH_SIZE * 2,
        shuffle            = False,
        num_workers        = num_workers,
        pin_memory         = (DEVICE.type == "cuda"),
        persistent_workers = (num_workers > 0),
    )

    # ── model ──
    model = SignClassifier(
        input_size  = FEATURE_DIM,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        num_classes = num_classes,
        dropout     = DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"\nModel parameters : {total_params:,}")
    log.info(f"Num classes      : {num_classes}")
    log.info(f"Hidden size      : {HIDDEN_SIZE} (smaller to reduce overfit)")
    log.info(f"Dropout          : {DROPOUT} (higher to reduce overfit)")
    log.info(f"Batch size       : {BATCH_SIZE}")
    log.info(f"Epochs           : {EPOCHS}")
    log.info(f"Patience         : {PATIENCE}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    scaler    = torch.amp.GradScaler("cuda")

    best_val_top1 = 0.0
    no_improve    = 0
    history       = []
    best_ckpt     = Path(MODEL_DIR) / "best_model.pth"

    log.info("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr1, tr5 = run_epoch(model, train_loader, criterion, optimizer, scaler)
        va_loss, va1, va5 = run_epoch(model, val_loader, criterion)

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        log.info(
            f"Epoch {epoch:03d}/{EPOCHS}  "
            f"train_loss={tr_loss:.4f}  train_top1={tr1:.1f}%  "
            f"val_loss={va_loss:.4f}  val_top1={va1:.1f}%  "
            f"val_top5={va5:.1f}%  "
            f"lr={lr_now:.2e}  time={elapsed:.0f}s"
        )

        history.append({
            "epoch":   epoch,
            "tr_loss": round(tr_loss, 4),
            "tr_top1": round(tr1, 2),
            "va_loss": round(va_loss, 4),
            "va_top1": round(va1, 2),
            "va_top5": round(va5, 2),
        })

        if va1 > best_val_top1:
            best_val_top1 = va1
            no_improve    = 0
            torch.save({
                "epoch":       epoch,
                "val_top1":    va1,
                "val_top5":    va5,
                "model_state": model.state_dict(),
                "label_remap": label_remap,
                "remap_inv":   remap_inv,
                "config": {
                    "input_size":  FEATURE_DIM,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers":  NUM_LAYERS,
                    "num_classes": num_classes,
                    "dropout":     DROPOUT,
                    "seq_len":     SEQ_LEN,
                },
            }, best_ckpt)
            log.info(f"  ✓ New best val_top1={va1:.2f}%  saved → best_model.pth")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                log.info(f"\nEarly stopping — no improvement for {PATIENCE} epochs.")
                break

        (Path(MODEL_DIR) / "training_log.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")

    # ── save label encoder ──
    log.info("\nSaving label encoder...")
    if Path(CLASS_MAP).exists():
        cm = json.loads(Path(CLASS_MAP).read_text(encoding="utf-8"))
        label_enc = {
            str(new): cm.get(str(orig), f"sign_{orig}")
            for orig, new in label_remap.items()
        }
    else:
        label_enc = {str(new): str(orig) for orig, new in label_remap.items()}

    enc_path = Path(MODEL_DIR) / "label_encoder.json"
    enc_path.write_text(json.dumps(label_enc, indent=2), encoding="utf-8")

    log.info(f"\n{'='*60}")
    log.info("TRAINING COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"Best val top-1 : {best_val_top1:.2f}%")
    log.info(f"Model saved    : {best_ckpt}")
    log.info(f"Label encoder  : {enc_path}")
    log.info(f"\nNext step: python main.py")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()