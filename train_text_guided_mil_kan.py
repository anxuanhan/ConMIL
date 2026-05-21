"""
train_text_guided_mil_kan.py
----------------------------
TextGuidedMIL_KAN 训练脚本

与 train_text_guided_mil.py 完全一致，唯一区别：
  - 导入 build_text_guided_mil_kan 代替 build_text_guided_mil
  - 多一个 --kan_grid 参数（B-spline 控制点数，默认5）

用法：
  # POS only
  python train_text_guided_mil_kan.py \
      --feat_dir      /path/to/features_conch \
      --label_csv     /path/to/reference.csv \
      --text_feat_dir /path/to/text_features \
      --use_neg false \
      --epochs 50 \
      --output ./results/kan

  # POS + NEG
  python train_text_guided_mil_kan.py \
      --feat_dir      /path/to/features_conch \
      --label_csv     /path/to/reference.csv \
      --text_feat_dir /path/to/text_features \
      --use_neg true \
      --neg_weight 0.5 \
      --kan_grid 5 \
      --epochs 50 \
      --output ./results/kan
"""

import os
import json
import argparse
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    average_precision_score, confusion_matrix,
)

from text_guided_mil_kan import build_text_guided_mil_kan


# ════════════════════════════════════════════════════════════
# 1. Dataset
# ════════════════════════════════════════════════════════════
class BagDataset(Dataset):
    MAX_PATCHES = 4096

    def __init__(self, feat_dir: str, df: pd.DataFrame, training: bool = False):
        import h5py
        self.training = training
        self.feats    = []
        self.labels   = []
        self.names    = []
        missing       = []

        for _, row in df.iterrows():
            stem  = os.path.splitext(row["image"])[0]
            h5    = os.path.join(feat_dir, f"{stem}.h5")
            label = 1 if row["type"] == "tumor" else 0

            if os.path.exists(h5):
                with h5py.File(h5, "r") as f:
                    feat = torch.from_numpy(f["features"][:]).float()
                self.feats.append(feat)
                self.labels.append(label)
                self.names.append(stem)
            else:
                missing.append(stem)

        if missing:
            print(f"  ⚠️  {len(missing)} h5 not found: {missing[:5]}"
                  f"{'...' if len(missing) > 5 else ''}")

        patch_counts = [f.shape[0] for f in self.feats]
        print(f"  📦 Loaded {len(self.feats)} bags  |  "
              f"Patches/slide → min={min(patch_counts)}, "
              f"max={max(patch_counts)}, "
              f"mean={int(np.mean(patch_counts))}, "
              f"median={int(np.median(patch_counts))}")

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        feat  = self.feats[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if (self.training and self.MAX_PATCHES is not None
                and feat.shape[0] > self.MAX_PATCHES):
            perm = torch.randperm(feat.shape[0])[:self.MAX_PATCHES]
            feat = feat[perm]
        return feat, label


def collate_fn(batch):
    feats, labels = zip(*batch)
    return list(feats), torch.stack(labels)


def make_loader(dataset, shuffle, num_workers):
    return DataLoader(
        dataset, batch_size=1, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
    )


# ════════════════════════════════════════════════════════════
# 2. 加载文字特征
# ════════════════════════════════════════════════════════════
def load_text_features(text_feat_dir, device):
    pos_path = os.path.join(text_feat_dir, "pos_features.npy")
    neg_path = os.path.join(text_feat_dir, "neg_features.npy")

    if not os.path.exists(pos_path):
        raise FileNotFoundError(
            f"pos_features.npy not found in {text_feat_dir}\n"
            f"Please run encode_text_queries.py first.")

    pos_feats = torch.from_numpy(
        np.load(pos_path).astype(np.float32)).to(device)

    neg_feats = None
    if os.path.exists(neg_path):
        neg_feats = torch.from_numpy(
            np.load(neg_path).astype(np.float32)).to(device)

    print(f"📂 Text features loaded:")
    print(f"   pos: {pos_feats.shape}")
    print(f"   neg: {neg_feats.shape if neg_feats is not None else 'not found'}")

    return pos_feats, neg_feats


# ════════════════════════════════════════════════════════════
# 3. Metrics
# ════════════════════════════════════════════════════════════
def compute_metrics(labels, probs, preds) -> dict:
    auc  = roc_auc_score(labels, probs)
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, zero_division=0)
    ap   = average_precision_score(labels, probs)
    cm   = confusion_matrix(labels, preds, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = cm[0, 0], 0, 0, cm[-1, -1]
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    return dict(AUC=float(auc), ACC=float(acc),
                Sensitivity=float(sens), Specificity=float(spec),
                F1=float(f1), AP=float(ap))


# ════════════════════════════════════════════════════════════
# 4. Train / Eval one epoch
# ════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, device,
                pos_feats, neg_feats) -> float:
    model.train()
    total_loss = 0.0

    for feats_list, labels in loader:
        x     = feats_list[0].to(device, non_blocking=True)
        label = labels[0].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits, _, _ = model(x, pos_feats, neg_feats)
        loss = nn.CrossEntropyLoss()(logits, label.unsqueeze(0))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device, pos_feats, neg_feats) -> dict:
    model.eval()
    all_labels, all_probs, all_preds = [], [], []

    for feats_list, labels in loader:
        x     = feats_list[0].to(device, non_blocking=True)
        label = labels[0].item()

        logits, _, _ = model(x, pos_feats, neg_feats)
        prob = torch.softmax(logits, dim=-1)[0, 1].item()
        pred = int(logits.argmax(dim=-1).item())

        all_labels.append(label)
        all_probs.append(prob)
        all_preds.append(pred)

    return compute_metrics(
        np.array(all_labels),
        np.array(all_probs),
        np.array(all_preds),
    )


# ════════════════════════════════════════════════════════════
# 5. Training loop
# ════════════════════════════════════════════════════════════
def train(args, train_df, val_df, device, pos_feats, neg_feats):
    num_workers = min(args.num_workers, os.cpu_count() or 1)

    print("\n📂 Loading training bags...")
    train_ds = BagDataset(args.feat_dir, train_df, training=True)
    print("\n📂 Loading validation bags...")
    val_ds   = BagDataset(args.feat_dir, val_df,   training=False)

    train_dl = make_loader(train_ds, shuffle=True,  num_workers=num_workers)
    val_dl   = make_loader(val_ds,   shuffle=False, num_workers=num_workers)

    model = build_text_guided_mil_kan(
        in_dim     = args.in_dim,
        hidden_dim = args.hidden_dim,
        n_classes  = 2,
        dropout    = args.dropout,
        alpha_init = args.alpha_init,
        use_neg    = args.use_neg,
        neg_weight = args.neg_weight,
        kan_grid   = args.kan_grid,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mode_str = "POS+NEG" if args.use_neg else "POS only"
    print(f"\n{'═'*65}")
    print(f"  TextGuidedMIL-KAN  [{mode_str}]")
    print(f"  Parameters  : {n_params:,}")
    print(f"  kan_grid    : {args.kan_grid}")
    print(f"  alpha_init  : {args.alpha_init}  "
          f"neg_weight : {args.neg_weight if args.use_neg else 'N/A'}")
    print(f"  train={len(train_ds)}  val={len(val_ds)}  epochs={args.epochs}")
    print(f"{'═'*65}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc  = 0.0
    run_name  = f"textguided_kan_{'posneg' if args.use_neg else 'pos'}"
    best_ckpt = os.path.join(args.output, f"{run_name}_best.pt")
    history   = []

    for epoch in range(1, args.epochs + 1):
        train_loss  = train_epoch(model, train_dl, optimizer, device,
                                  pos_feats, neg_feats)
        val_metrics = eval_epoch(model, val_dl, device, pos_feats, neg_feats)
        scheduler.step()

        alpha_val = model.alpha.item()

        is_best = val_metrics["ACC"] > best_acc
        if is_best:
            best_acc = val_metrics["ACC"]
            torch.save(model.state_dict(), best_ckpt)

        history.append({
            "epoch"     : epoch,
            "train_loss": round(train_loss, 6),
            "alpha"     : round(alpha_val, 6),
            **{k: round(v, 6) for k, v in val_metrics.items()},
            "is_best"   : is_best,
        })

        if epoch % 5 == 0 or epoch == 1 or is_best:
            flag = "  ← best ✓" if is_best else ""
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"loss={train_loss:.4f}  "
                  f"alpha={alpha_val:.3f}  "
                  f"val_ACC={val_metrics['ACC']:.4f}  "
                  f"val_AUC={val_metrics['AUC']:.4f}  "
                  f"val_F1={val_metrics['F1']:.4f}"
                  f"{flag}")

    print(f"\n  ✅ Best val ACC = {best_acc:.4f}  →  {best_ckpt}")

    hist_path = os.path.join(args.output, f"{run_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  📈 History saved → {hist_path}")

    return best_ckpt, run_name


# ════════════════════════════════════════════════════════════
# 6. Test
# ════════════════════════════════════════════════════════════
def test(args, test_df, ckpt_path, run_name, device, pos_feats, neg_feats):
    num_workers = min(args.num_workers, os.cpu_count() or 1)

    print("\n📂 Loading test bags...")
    test_ds = BagDataset(args.feat_dir, test_df, training=False)
    test_dl = make_loader(test_ds, shuffle=False, num_workers=num_workers)

    model = build_text_guided_mil_kan(
        in_dim     = args.in_dim,
        hidden_dim = args.hidden_dim,
        n_classes  = 2,
        dropout    = args.dropout,
        alpha_init = args.alpha_init,
        use_neg    = args.use_neg,
        neg_weight = args.neg_weight,
        kan_grid   = args.kan_grid,
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"  🔄 Loaded: {ckpt_path}")
    print(f"  Learned alpha = {model.alpha.item():.4f}  "
          f"(初始值={args.alpha_init})")

    metrics = eval_epoch(model, test_dl, device, pos_feats, neg_feats)

    mode_str = "POS+NEG" if args.use_neg else "POS only"
    print(f"\n{'═'*65}")
    print(f"  ✨ Test Results  —  TextGuidedMIL-KAN [{mode_str}]")
    print(f"{'═'*65}")
    for k, v in metrics.items():
        print(f"  {k:<16}: {v:.4f}")
    print(f"{'═'*65}\n")

    return metrics


# ════════════════════════════════════════════════════════════
# 7. Args & Main
# ════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data
    p.add_argument("--feat_dir",      required=True)
    p.add_argument("--label_csv",     required=True)
    p.add_argument("--text_feat_dir", required=True)

    # Model
    p.add_argument("--in_dim",      type=int,   default=512)
    p.add_argument("--hidden_dim",  type=int,   default=256)
    p.add_argument("--dropout",     type=float, default=0.25)
    p.add_argument("--alpha_init",  type=float, default=0.5)
    p.add_argument("--kan_grid",    type=int,   default=5,
                   help="KAN B-spline 控制点数（5~10，越大越精细但越慢）")

    # Text guidance
    p.add_argument("--use_neg",    type=lambda x: x.lower() == "true",
                   default=False)
    p.add_argument("--neg_weight", type=float, default=0.5)

    # Training
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--val_ratio",    type=float, default=0.2)
    p.add_argument("--max_patches",  type=int,   default=4096)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--output",       default="./results/kan")

    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    BagDataset.MAX_PATCHES = None if args.max_patches <= 0 else args.max_patches

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_str = "POS+NEG" if args.use_neg else "POS only"

    print(f"Device        : {device}")
    print(f"Mode          : {mode_str}")
    print(f"kan_grid      : {args.kan_grid}")
    print(f"feat_dir      : {args.feat_dir}")
    print(f"text_feat_dir : {args.text_feat_dir}")

    pos_feats, neg_feats = load_text_features(args.text_feat_dir, device)
    if args.use_neg and neg_feats is None:
        raise FileNotFoundError(
            "use_neg=True 但找不到 neg_features.npy，请先运行 encode_text_queries.py")

    # 读 CSV，过滤无 h5 的行
    df = pd.read_csv(args.label_csv)
    def h5_exists(img):
        stem = os.path.splitext(img)[0]
        return os.path.exists(os.path.join(args.feat_dir, f"{stem}.h5"))
    before = len(df)
    df = df[df["image"].apply(h5_exists)].reset_index(drop=True)
    if len(df) < before:
        print(f"  ⚠️  Skipped {before - len(df)} rows (h5 not found)")

    # train / test 分割
    is_test  = df["image"].str.startswith("test_")
    train_df = df[~is_test].reset_index(drop=True)
    test_df  = df[is_test].reset_index(drop=True)

    print(f"\n  Train slides : {len(train_df)}"
          f"  ({(train_df['type']=='tumor').sum()} tumor"
          f" / {(train_df['type']=='normal').sum()} normal)")
    print(f"  Test  slides : {len(test_df)}"
          f"  ({(test_df['type']=='tumor').sum()} tumor"
          f" / {(test_df['type']=='normal').sum()} normal)")

    if len(test_df) == 0:
        raise RuntimeError("❌ No test slides found.")

    # train → train + val
    train_labels = (train_df["type"] == "tumor").astype(int).values
    tr_idx, val_idx = train_test_split(
        range(len(train_df)),
        test_size=args.val_ratio,
        stratify=train_labels,
        random_state=args.seed,
    )
    pure_train_df = train_df.iloc[list(tr_idx)].reset_index(drop=True)
    val_df        = train_df.iloc[list(val_idx)].reset_index(drop=True)

    print(f"\n  Train → train : {len(pure_train_df)} slides")
    print(f"  Train → val   : {len(val_df)} slides")

    # 训练
    best_ckpt, run_name = train(
        args, pure_train_df, val_df, device, pos_feats, neg_feats
    )

    # 测试
    test_metrics = test(
        args, test_df, best_ckpt, run_name, device, pos_feats, neg_feats
    )

    # 保存结果
    result = {
        "model"        : f"TextGuidedMIL_KAN_{mode_str}",
        "use_neg"      : args.use_neg,
        "neg_weight"   : args.neg_weight if args.use_neg else None,
        "alpha_init"   : args.alpha_init,
        "kan_grid"     : args.kan_grid,
        "in_dim"       : args.in_dim,
        "hidden_dim"   : args.hidden_dim,
        "epochs"       : args.epochs,
        "lr"           : args.lr,
        "weight_decay" : args.weight_decay,
        "seed"         : args.seed,
        "n_train"      : len(pure_train_df),
        "n_val"        : len(val_df),
        "n_test"       : len(test_df),
        "best_ckpt"    : best_ckpt,
        "test_metrics" : test_metrics,
    }
    out_path = os.path.join(args.output, f"{run_name}_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()