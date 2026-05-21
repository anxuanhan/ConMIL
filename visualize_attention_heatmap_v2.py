"""
visualize_attention_heatmap_v2.py
----------------------------------
在原版基础上的改进：
  1. 颜色标准化：用百分位裁剪 + 幂次变换增强对比度，低注意力区域不再抢色
  2. Mask 叠加：读取 WSI 对应 mask，背景区域不渲染热图，保持白色/透明

用法：
  python visualize_attention_heatmap_v2.py \
      --slide       /path/to/tumor_003.tif \
      --mask        /path/to/tumor_003_mask.tif \
      --h5          /path/to/tumor_003.h5 \
      --ckpt        /path/to/model.pt \
      --text_feat_dir /path/to/text_features \
      --output      ./tumor_003_heatmap_v2.png

新增参数：
  --mask          对应 WSI 的 mask .tif 路径（留空则不使用 mask）
  --low_pct       1.0    attention 归一化下界百分位（裁掉底部噪声）
  --high_pct      99.0   attention 归一化上界百分位（裁掉顶部离群值）
  --gamma         0.5    幂次变换指数（<1 拉伸低值区, >1 压缩低值区）
"""

import argparse
import os
import math
import numpy as np
import h5py
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from PIL import Image
import tifffile
from PIL import Image
import numpy as np

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False
    print("⚠️  openslide 未安装，将使用 tifffile 作为备选读取器")
    try:
        import tifffile
        HAS_TIFFFILE = True
    except ImportError:
        HAS_TIFFFILE = False

import torch.nn as nn


# ════════════════════════════════════════════════════════════
# 1. 模型定义（与原版完全一致）
# ════════════════════════════════════════════════════════════
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features,
                 grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 grid_range=(-1.0, 1.0)):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float)
            * h + grid_range[0]
        )
        grid = grid.unsqueeze(0).expand(in_features, -1)
        self.register_buffer("grid", grid)
        self.base_weight   = nn.Parameter(torch.empty(out_features, in_features))
        n_basis = grid_size + spline_order
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, n_basis))
        self.scale_base    = scale_base
        self.scale_spline  = scale_spline
        nn.init.kaiming_uniform_(self.base_weight, a=5**0.5)
        with torch.no_grad():
            noise = (torch.rand(self.spline_weight.shape)*2-1)*scale_noise
            self.spline_weight.copy_(noise)

    def b_splines(self, x):
        *batch, D = x.shape
        x = x.reshape(-1, D).unsqueeze(-1)
        grid = self.grid
        basis = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()
        for k in range(1, self.spline_order + 1):
            d_left  = grid[:, k:-1]   - grid[:, :-(k+1)]
            d_right = grid[:, k+1:]   - grid[:, 1:-k]
            left  = (x - grid[:, :-(k+1)]) / (d_left  + 1e-8) * basis[:, :, :-1]
            right = (grid[:, k+1:] - x)    / (d_right + 1e-8) * basis[:, :,  1:]
            basis = left + right
        return basis.reshape(*batch, D, -1)

    def forward(self, x):
        base_out = F.linear(F.silu(x), self.base_weight * self.scale_base)
        B = self.b_splines(x)
        *batch, D, nb = B.shape
        B_flat = B.reshape(-1, D * nb)
        W_flat = self.spline_weight.reshape(self.out_features, D * nb) * self.scale_spline
        spline_out = F.linear(B_flat, W_flat).reshape(*batch, self.out_features)
        return base_out + spline_out


class KAN(nn.Module):
    def __init__(self, dims, grid_size=5, dropout=0.25):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(KANLinear(dims[i], dims[i+1], grid_size=grid_size))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TextGuidedMIL_KAN(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, n_classes=2,
                 dropout=0.25, alpha_init=0.5,
                 use_neg=False, neg_weight=0.5, kan_grid=5):
        super().__init__()
        self.use_neg    = use_neg
        self.neg_weight = neg_weight
        self.alpha      = nn.Parameter(torch.tensor(alpha_init))
        self.feat_proj  = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attention_V = nn.Linear(hidden_dim, 128)
        self.attention_U = nn.Linear(hidden_dim, 128)
        self.attention_W = nn.Linear(128, 1)
        self.sim_proj    = nn.Linear(1, 1, bias=False)
        self.classifier  = KAN(
            dims=[hidden_dim, hidden_dim//2, n_classes],
            grid_size=kan_grid, dropout=dropout)

    def compute_text_sim(self, patch_feats, pos_feats, neg_feats=None):
        pos_sim = (patch_feats @ pos_feats.T).mean(dim=1, keepdim=True)
        if self.use_neg and neg_feats is not None:
            neg_sim = (patch_feats @ neg_feats.T).mean(dim=1, keepdim=True)
            return pos_sim - self.neg_weight * neg_sim
        return pos_sim

    def forward(self, patch_feats, pos_feats, neg_feats=None):
        sim_score    = self.compute_text_sim(patch_feats, pos_feats, neg_feats)
        h            = self.feat_proj(patch_feats)
        A_V          = torch.tanh(self.attention_V(h))
        A_U          = torch.sigmoid(self.attention_U(h))
        A            = self.attention_W(A_V * A_U)
        text_bias    = self.alpha * self.sim_proj(sim_score)
        A_guided     = A + text_bias
        attn_weights = F.softmax(A_guided, dim=0)
        bag          = (attn_weights * h).sum(dim=0, keepdim=True)
        logits       = self.classifier(bag)
        return logits, attn_weights.squeeze(), sim_score.squeeze()


# ════════════════════════════════════════════════════════════
# 2. WSI / Mask 读取工具
# ════════════════════════════════════════════════════════════
def read_wsi_level(slide_path, vis_level):
    """返回 (np.ndarray H×W×3, downsample_factor)"""
    if HAS_OPENSLIDE:
        slide      = openslide.OpenSlide(slide_path)
        level      = min(vis_level, len(slide.level_dimensions) - 1)
        dims       = slide.level_dimensions[level]
        thumb      = np.array(slide.read_region((0, 0), level, dims).convert("RGB"))
        downsample = slide.level_downsamples[level]
        slide.close()
        return thumb, downsample
    else:
        import tifffile
        with tifffile.TiffFile(slide_path) as tif:
            series = tif.series[0]
            levels = getattr(series, 'levels', [series])
            level  = min(vis_level, len(levels) - 1)
            arr    = levels[level].asarray()
        if arr.ndim == 3 and arr.shape[0] in [3, 4]:
            arr = arr.transpose(1, 2, 0)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        downsample = 2 ** level
        return arr.astype(np.uint8), float(downsample)


def read_mask_binary(mask_path, target_hw, vis_level):
    """
    读取 mask tif，resize 到与 tissue thumbnail 相同大小，返回 bool (H, W)。
    target_hw: (H, W)
    """
    H, W = target_hw
 
    import tifffile
    with tifffile.TiffFile(mask_path) as tif:
        series = tif.series[0]
        levels = getattr(series, 'levels', [series])
        level  = min(vis_level, len(levels) - 1)
        arr    = levels[level].asarray()
    if arr.ndim == 3:
        arr = arr[..., 0]

    # Resize to match tissue thumbnail
    if arr.shape[:2] != (H, W):
        arr = np.array(
            Image.fromarray(arr.astype(np.uint8)).resize((W, H), resample=Image.NEAREST)
        )
    return arr > 0


# ════════════════════════════════════════════════════════════
# 3. 颜色标准化（核心改进）
# ════════════════════════════════════════════════════════════
def normalize_attention(attn, low_pct=1.0, high_pct=99.0, gamma=0.5):
    """
    百分位裁剪 + 幂次变换，让颜色分布更均匀。

    low_pct  : 裁掉底部百分位（清除接近零的噪声）
    high_pct : 裁掉顶部百分位（防止极值拉平其他颜色）
    gamma    : 幂次，<1 则低注意力区域颜色被拉伸放大
    """
    lo = np.percentile(attn, low_pct)
    hi = np.percentile(attn, high_pct)
    attn_clipped = np.clip(attn, lo, hi)
    attn_norm    = (attn_clipped - lo) / (hi - lo + 1e-8)
    attn_gamma   = np.power(attn_norm, gamma)   # gamma < 1 → 拉伸低值
    return attn_gamma.astype(np.float32)


# ════════════════════════════════════════════════════════════
# 4. 绘制 heatmap（含 mask）
# ════════════════════════════════════════════════════════════
def draw_heatmap(
    slide_path, mask_path, coords, attn_weights, sim_scores,
    patch_size, vis_level, output_path,
    alpha=0.5, cmap="RdBu_r",
    low_pct=1.0, high_pct=99.0, gamma=0.5,
    pred_prob=None, patch_ref_level=2,
):
    print(f"📖 读取 WSI thumbnail (level={vis_level})...")
    tissue_np, downsample = read_wsi_level(slide_path, vis_level)
    H, W = tissue_np.shape[:2]
    print(f"   Thumbnail size: {W} × {H}  (downsample={downsample:.1f}x)")

    # ── 读取并对齐 Mask ────────────────────────────────────
    tissue_mask = None  # True = 组织区域
    if mask_path and os.path.exists(mask_path):
        print(f"📖 读取 Mask: {mask_path}")
        tissue_mask = read_mask_binary(mask_path, (H, W), vis_level)
        cov = tissue_mask.mean() * 100
        print(f"   Mask 组织覆盖率: {cov:.1f}%")
    else:
        print("   ⚠️  未提供 mask，将显示全图")

    # ── 改进的颜色归一化 ───────────────────────────────────
    attn_raw  = attn_weights.cpu().numpy().astype(np.float32)
    attn_norm = normalize_attention(attn_raw, low_pct, high_pct, gamma)
    print(f"   Attention 归一化后 — min={attn_norm.min():.4f}  "
          f"max={attn_norm.max():.4f}  mean={attn_norm.mean():.4f}")

    # ── 推算 patch 在可视化层的大小 ───────────────────────
    if HAS_OPENSLIDE:
        slide_meta     = openslide.OpenSlide(slide_path)
        ref_level      = min(patch_ref_level, len(slide_meta.level_downsamples) - 1)
        ref_downsample = float(slide_meta.level_downsamples[ref_level])
        slide_meta.close()
    else:
        ref_downsample = float(2 ** patch_ref_level)
    ps_vis = max(int(round(patch_size * ref_downsample / float(downsample))), 1)

    # ── 建立 heatmap canvas ────────────────────────────────
    heatmap   = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for i, (x, y) in enumerate(coords):
        x_vis = int(x / downsample)
        y_vis = int(y / downsample)
        x_end = min(x_vis + ps_vis, W)
        y_end = min(y_vis + ps_vis, H)
        if x_vis >= W or y_vis >= H:
            continue
        heatmap[y_vis:y_end, x_vis:x_end]   += attn_norm[i]
        count_map[y_vis:y_end, x_vis:x_end] += 1.0

    valid = count_map > 0
    heatmap[valid] /= count_map[valid]

    # ── 颜色映射 ───────────────────────────────────────────
    colormap     = cm.get_cmap(cmap)
    norm_obj     = Normalize(vmin=0, vmax=1)
    heatmap_rgba = colormap(norm_obj(heatmap))
    heatmap_rgb  = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    # ── patch 区域 mask（有 patch 的位置）──────────────────
    patch_mask = (count_map > 0)

    # ── 若提供 tissue mask，仅在组织区域渲染 ──────────────
    if tissue_mask is not None:
        render_mask = patch_mask & tissue_mask
    else:
        render_mask = patch_mask

    render_f = render_mask.astype(np.float32)

    # ── Panel A: 原图（mask 外变白）──────────────────────
    tissue_masked = tissue_np.copy().astype(np.float32)
    if tissue_mask is not None:
        for c in range(3):
            tissue_masked[:, :, c] = np.where(tissue_mask, tissue_np[:, :, c], 255)
    tissue_masked = tissue_masked.clip(0, 255).astype(np.uint8)

    # ── Panel B: overlay（背景白 + 热图叠加在 render_mask）
    overlay = np.full_like(tissue_np, 255, dtype=np.float32)
    if tissue_mask is not None:
        # 组织区域先填原图
        for c in range(3):
            overlay[:, :, c] = np.where(tissue_mask,
                                         tissue_np[:, :, c].astype(np.float32), 255.0)
    else:
        overlay = tissue_np.copy().astype(np.float32)

    for c in range(3):
        overlay[:, :, c] = (
            overlay[:, :, c] * (1 - render_f * alpha)
            + heatmap_rgb[:, :, c] * (render_f * alpha)
        )
    overlay = overlay.clip(0, 255).astype(np.uint8)

    # ── Panel C: 纯热图（白底 + 组织区内颜色块）──────────
    bg = np.full((H, W, 3), 255, dtype=np.uint8)
    for c in range(3):
        bg[:, :, c] = np.where(render_mask, heatmap_rgb[:, :, c], bg[:, :, c])

    # ── 绘图 ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=150)
    fig.patch.set_facecolor("white")

    panel_titles = [
        "Tissue (mask applied)",
        "Attention Overlay (tissue only)",
        "Pure Attention Map",
    ]
    imgs = [tissue_masked, overlay, bg]

    for i, (ax, title, img) in enumerate(zip(axes, panel_titles, imgs)):
        ax.set_facecolor("black")
        ax.axis("off")
        ax.set_title(title, color="white", fontsize=13, pad=8, fontweight="bold")
        ax.imshow(img)

    # ── Colorbar ───────────────────────────────────────────
    # ── Colorbar（绑定第二张图） ─────────────────────────
    sm = cm.ScalarMappable(cmap=cmap, norm=norm_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.03, pad=0.02)

    cbar.set_label(
        f"Attention (p{low_pct:.0f}–p{high_pct:.0f}, γ={gamma})",
        color="black", fontsize=10
    )
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")




    sm = cm.ScalarMappable(cmap=cmap, norm=norm_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[2], fraction=0.03, pad=0.02)
    cbar.set_label(
        f"Attention (p{low_pct:.0f}–p{high_pct:.0f}, γ={gamma})",
        color="black", fontsize=10
    )
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")

    # ── 统计信息 ───────────────────────────────────────────
    slide_name = os.path.basename(slide_path)
    lo_val = np.percentile(attn_raw, low_pct)
    hi_val = np.percentile(attn_raw, high_pct)

    info_lines = [
        f"Slide: {slide_name}",
        f"Mask: {'yes' if tissue_mask is not None else 'no'}",
        f"Patches: {len(coords):,}",
        f"Patch size: L{patch_ref_level}:{patch_size}px → vis L{vis_level}: {ps_vis}px",
        f"Attn raw — max:{attn_raw.max():.5f}  mean:{attn_raw.mean():.5f}",
        f"Norm clip: [{lo_val:.5f}, {hi_val:.5f}]  γ={gamma}",
    ]
    if pred_prob is not None:
        info_lines.append(f"Pred prob (tumor): {pred_prob:.4f}")

    fig.text(0.01, 0.02, "\n".join(info_lines),
             color="lightgray", fontsize=9, va="bottom", ha="left",
             bbox=dict(facecolor="#333355", alpha=0.6, edgecolor="none", pad=4))

    plt.suptitle(
        f"Attention Heatmap — TextGuidedMIL-KAN  (v2)\n{slide_name}",
        color="black", fontsize=15, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=150)
    plt.close()
    print(f"✅ Heatmap saved → {output_path}")


# ════════════════════════════════════════════════════════════
# 5. Top-K patch 预览（与原版一致）
# ════════════════════════════════════════════════════════════
def save_top_patches(slide_path, coords, attn_weights, patch_size,
                     output_dir, top_k=10):
    if not HAS_OPENSLIDE:
        print("  ⚠️  需要 openslide 才能保存 top patches，跳过")
        return
    os.makedirs(output_dir, exist_ok=True)
    attn    = attn_weights.cpu().numpy()
    top_idx = np.argsort(attn)[::-1][:top_k]
    slide   = openslide.OpenSlide(slide_path)
    ncols   = min(5, top_k)
    nrows   = int(math.ceil(top_k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    fig.patch.set_facecolor("#1a1a2e")
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])
    for rank, idx in enumerate(top_idx):
        x, y  = int(coords[idx][0]), int(coords[idx][1])
        patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
        axes[rank].imshow(np.array(patch))
        axes[rank].set_title(f"#{rank+1}  w={attn[idx]:.4f}", color="black", fontsize=8)
        axes[rank].axis("off")
        patch.save(os.path.join(output_dir, f"top_{rank+1:02d}_x{x}_y{y}.png"))
    for j in range(len(top_idx), len(axes)):
        axes[j].axis("off")
    slide.close()
    plt.suptitle(f"Top-{top_k} Highest Attention Patches",
                 color="black", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, f"top_{top_k}_patches_grid.png")
    plt.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=120)
    plt.close()
    print(f"✅ Top patches saved → {out}")

def save_colorbar_svg(cmap="RdBu_r", low_pct=1.0, high_pct=99.0, gamma=0.5):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(1.2, 4))  # 控制长条比例

    norm = Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)

    cbar.set_label(
        f"Attention (p{low_pct:.0f}–p{high_pct:.0f}, γ={gamma})",
        fontsize=10
    )

    ax.remove()  # 去掉多余轴

    plt.savefig("colorbar.svg", bbox_inches="tight", transparent=True)
    plt.close()



# ════════════════════════════════════════════════════════════
# 6. Main
# ════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--slide",         required=True,  help="WSI .tif 路径")
    p.add_argument("--mask",          default="",     help="Mask .tif 路径（留空则不使用）")
    p.add_argument("--h5",            required=True,  help=".h5 特征文件路径")
    p.add_argument("--ckpt",          required=True,  help="模型 checkpoint 路径")
    p.add_argument("--text_feat_dir", required=True,  help="pos/neg_features.npy 目录")
    p.add_argument("--output",        default="./heatmap_v2.png")

    # 颜色归一化（新增）
    p.add_argument("--low_pct",   type=float, default=1.0,
                   help="归一化下界百分位（清除底部噪声）")
    p.add_argument("--high_pct",  type=float, default=99.0,
                   help="归一化上界百分位（防止极值压平）")
    p.add_argument("--gamma",     type=float, default=0.5,
                   help="幂次变换 (<1 拉伸低注意力区颜色, >1 压缩)")

    # Top patches
    p.add_argument("--top_patches",      default="")
    p.add_argument("--top_k",            type=int, default=10)
    p.add_argument("--save_top_patches", type=lambda x: x.lower()=="true", default=True)

    # Visualization
    p.add_argument("--patch_size",      type=int,   default=256)
    p.add_argument("--patch_ref_level", type=int,   default=2)
    p.add_argument("--vis_level",       type=int,   default=4)
    p.add_argument("--alpha",           type=float, default=0.5)
    p.add_argument("--cmap",            default="RdBu_r")

    # Model
    p.add_argument("--in_dim",      type=int,   default=512)
    p.add_argument("--hidden_dim",  type=int,   default=256)
    p.add_argument("--kan_grid",    type=int,   default=6)
    p.add_argument("--dropout",     type=float, default=0.25)
    p.add_argument("--alpha_init",  type=float, default=0.5)
    p.add_argument("--use_neg",     type=lambda x: x.lower()=="true", default=True)
    p.add_argument("--neg_weight",  type=float, default=0.5)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载特征
    print(f"📂 Loading features: {args.h5}")
    with h5py.File(args.h5, "r") as f:
        feats  = torch.from_numpy(f["features"][:]).float()
        coords = f["coordinates"][:]
    print(f"   Features: {feats.shape}   Coords: {coords.shape}")
    feats = F.normalize(feats, dim=-1).to(device)

    # 加载文字特征
    pos_path  = os.path.join(args.text_feat_dir, "pos_features.npy")
    neg_path  = os.path.join(args.text_feat_dir, "neg_features.npy")
    pos_feats = torch.from_numpy(np.load(pos_path).astype(np.float32)).to(device)
    neg_feats = None
    if os.path.exists(neg_path):
        neg_feats = torch.from_numpy(np.load(neg_path).astype(np.float32)).to(device)
        print(f"   pos_feats: {pos_feats.shape}   neg_feats: {neg_feats.shape}")
    else:
        print(f"   pos_feats: {pos_feats.shape}   neg_feats: not found")

    # 加载模型
    print(f"🔄 Loading checkpoint: {args.ckpt}")
    model = TextGuidedMIL_KAN(
        in_dim=args.in_dim, hidden_dim=args.hidden_dim,
        n_classes=2, dropout=args.dropout,
        alpha_init=args.alpha_init, use_neg=args.use_neg,
        neg_weight=args.neg_weight, kan_grid=args.kan_grid,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f"   Learned alpha = {model.alpha.item():.4f}")

    # 推理
    with torch.no_grad():
        logits, attn_weights, sim_scores = model(feats, pos_feats, neg_feats)
        prob = torch.softmax(logits, dim=-1)[0, 1].item()
        pred = int(logits.argmax(dim=-1).item())

    label_str = "TUMOR" if pred == 1 else "NORMAL"
    print(f"\n🎯 Prediction: {label_str}  (tumor prob = {prob:.4f})")
    print(f"   Attention — min={attn_weights.min():.6f}  "
          f"max={attn_weights.max():.6f}  mean={attn_weights.mean():.6f}")
    print(f"   Norm config: low_pct={args.low_pct}  "
          f"high_pct={args.high_pct}  gamma={args.gamma}")

    # 绘图
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    draw_heatmap(
        slide_path     = args.slide,
        mask_path      = args.mask if args.mask else None,
        coords         = coords,
        attn_weights   = attn_weights,
        sim_scores     = sim_scores,
        patch_size     = args.patch_size,
        vis_level      = args.vis_level,
        output_path    = args.output,
        alpha          = args.alpha,
        cmap           = args.cmap,
        low_pct        = args.low_pct,
        high_pct       = args.high_pct,
        gamma          = args.gamma,
        pred_prob      = prob,
        patch_ref_level = args.patch_ref_level,
    )

    # Top patches
    if args.save_top_patches:
        top_patch_dir = args.top_patches
        if not top_patch_dir:
            base = os.path.splitext(os.path.basename(args.output))[0]
            top_patch_dir = os.path.join(
                os.path.dirname(os.path.abspath(args.output)),
                f"{base}_top{args.top_k}_patches"
            )
        save_top_patches(
            slide_path   = args.slide,
            coords       = coords,
            attn_weights = attn_weights,
            patch_size   = args.patch_size,
            output_dir   = top_patch_dir,
            top_k        = args.top_k,
        )

    save_colorbar_svg(
    cmap=args.cmap,
    low_pct=args.low_pct,
    high_pct=args.high_pct,
    gamma=args.gamma
)


if __name__ == "__main__":
    main()