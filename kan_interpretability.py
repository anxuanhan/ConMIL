"""
kan_interpretability.py
-----------------------
KAN 可解释性深度分析（面向审稿人的硬核证据）

分析内容：
  Part 1 — B-spline 曲线可视化
    · 提取 KAN 第一层（256→128）所有边的激活函数 φ_{j←i}(x)
    · 按振幅（amplitude）排序，绘制 Top-K 条曲线
    · 与等效 MLP 线性拟合做对比，直观展示非线性优势

  Part 2 — Feature Attribution（灵敏度分析）
    · 在真实测试集输入分布下，计算每条边的"能量"（振幅 × 激活频率）
    · 输出热力图：哪些 input_dim → hidden_dim 连接最具判别力
    · 区分肿瘤 vs 正常 patch 在这些关键边上的响应分布差异

  Part 3 — MLP vs KAN 非线性对比（量化）
    · 对每条 Top 边拟合一条线性回归，计算残差 R²
    · R² 越低，说明该连接越无法被线性函数覆盖 → KAN 的核心优势

用法（最简）：
  python kan_interpretability.py \
      --ckpt /path/to/textguided_kan6_posneg_best.pt \
      --output ./kan_analysis

用法（含真实数据 feature attribution）：
  python kan_interpretability.py \
      --ckpt        /path/to/textguided_kan6_posneg_best.pt \
      --h5_dir      /path/to/features_conch \
      --label_csv   /path/to/reference.csv \
      --text_feat_dir /path/to/text_features \
      --output      ./kan_analysis
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy import stats
from scipy.ndimage import gaussian_filter1d
plt.rcParams["svg.fonttype"] = "none"


# ════════════════════════════════════════════════════════════
# 0. 模型定义（与训练脚本完全一致）
# ════════════════════════════════════════════════════════════
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 grid_range=(-1.0, 1.0)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float)
                * h + grid_range[0])
        self.register_buffer("grid", grid.unsqueeze(0).expand(in_features, -1))
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        n_basis = grid_size + spline_order
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, n_basis))
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        nn.init.kaiming_uniform_(self.base_weight, a=5**0.5)
        with torch.no_grad():
            self.spline_weight.copy_((torch.rand(self.spline_weight.shape)*2-1)*scale_noise)

    def b_splines(self, x):
        *batch, D = x.shape
        x = x.reshape(-1, D).unsqueeze(-1)
        grid = self.grid
        basis = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()
        for k in range(1, self.spline_order + 1):
            d_left  = grid[:, k:-1]  - grid[:, :-(k+1)]
            d_right = grid[:, k+1:]  - grid[:, 1:-k]
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
        return base_out + F.linear(B_flat, W_flat).reshape(*batch, self.out_features)


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
                 dropout=0.25, alpha_init=0.5, use_neg=False,
                 neg_weight=0.5, kan_grid=5):
        super().__init__()
        self.use_neg = use_neg
        self.neg_weight = neg_weight
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.feat_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attention_V = nn.Linear(hidden_dim, 128)
        self.attention_U = nn.Linear(hidden_dim, 128)
        self.attention_W = nn.Linear(128, 1)
        self.sim_proj = nn.Linear(1, 1, bias=False)
        self.classifier = KAN(
            dims=[hidden_dim, hidden_dim//2, n_classes],
            grid_size=kan_grid, dropout=dropout)

    def compute_text_sim(self, patch_feats, pos_feats, neg_feats=None):
        pos_sim = (patch_feats @ pos_feats.T).mean(dim=1, keepdim=True)
        if self.use_neg and neg_feats is not None:
            neg_sim = (patch_feats @ neg_feats.T).mean(dim=1, keepdim=True)
            return pos_sim - self.neg_weight * neg_sim
        return pos_sim

    def forward(self, patch_feats, pos_feats, neg_feats=None):
        sim_score = self.compute_text_sim(patch_feats, pos_feats, neg_feats)
        h = self.feat_proj(patch_feats)
        A_V = torch.tanh(self.attention_V(h))
        A_U = torch.sigmoid(self.attention_U(h))
        A = self.attention_W(A_V * A_U)
        text_bias = self.alpha * self.sim_proj(sim_score)
        attn_weights = F.softmax(A + text_bias, dim=0)
        bag = (attn_weights * h).sum(dim=0, keepdim=True)
        logits = self.classifier(bag)
        return logits, attn_weights.squeeze(), sim_score.squeeze()


# ════════════════════════════════════════════════════════════
# 1. 核心工具：单条边的激活函数 φ_{j←i}(x)
# ════════════════════════════════════════════════════════════
def get_kan_layer(model):
    """返回 KAN 分类器的第一个 KANLinear 层（256→128）"""
    for m in model.classifier.net:
        if isinstance(m, KANLinear):
            return m
    raise ValueError("找不到 KANLinear 层")


def get_kan_layer2(model):
    """返回 KAN 分类器的第二个 KANLinear 层（128→2）"""
    found = []
    for m in model.classifier.net:
        if isinstance(m, KANLinear):
            found.append(m)
    return found[1] if len(found) > 1 else None


def eval_single_edge(layer: KANLinear, in_idx: int, out_idx: int,
                     x_vals: np.ndarray) -> np.ndarray:
    """
    计算第 (out_idx, in_idx) 条边的激活函数 φ(x) 在 x_vals 上的值。
    φ(x) = scale_base * base_weight[out, in] * SiLU(x)
           + scale_spline * Σ_k spline_weight[out, in, k] * B_k(x)
    """
    device = layer.base_weight.device
    x_t = torch.tensor(x_vals, dtype=torch.float32, device=device).unsqueeze(1)
    # 扩展为 (N, in_features) — 其余维度填0，只激活 in_idx
    x_full = torch.zeros(len(x_vals), layer.in_features, device=device)
    x_full[:, in_idx] = x_t.squeeze(1)

    with torch.no_grad():
        B = layer.b_splines(x_full)                       # (N, in, n_basis)
        spline_vals = (B[:, in_idx, :] *                  # (N, n_basis)
                       layer.spline_weight[out_idx, in_idx, :].unsqueeze(0)  # (1, n_basis)
                       ).sum(dim=1) * layer.scale_spline   # (N,)

        silu_x = F.silu(x_t.squeeze(1))
        base_vals = silu_x * layer.base_weight[out_idx, in_idx] * layer.scale_base

    phi = (base_vals + spline_vals).cpu().numpy()
    return phi


def compute_amplitude_matrix(layer: KANLinear, x_vals: np.ndarray,
                              max_pairs: int = 5000) -> np.ndarray:
    """
    计算所有 (out, in) 边的振幅矩阵 amp[out, in] = max(φ) - min(φ)。
    若维度太大，只随机采样 max_pairs 条边计算（其余填0）。
    返回 (out_features, in_features)
    """
    OUT, IN = layer.out_features, layer.in_features
    amp_mat = np.zeros((OUT, IN), dtype=np.float32)

    device = layer.base_weight.device
    x_t = torch.tensor(x_vals, dtype=torch.float32, device=device)  # (N,)

    # 预计算所有节点的 B-spline 基（在1D网格上，只需遍历in_features）
    x_col = x_t.unsqueeze(1).expand(-1, IN)  # (N, IN) — 每列都是同一序列

    with torch.no_grad():
        x_full = torch.zeros(len(x_vals), IN, device=device)
        for i in range(IN):
            x_full[:, i] = x_t

        B_all = layer.b_splines(x_full)  # (N, IN, n_basis)
        # SiLU(x) 是 scalar 函数，对每个 in_idx 独立
        silu_all = F.silu(x_t).unsqueeze(1).expand(-1, IN)  # (N, IN)

    # 逐 out_idx 计算
    with torch.no_grad():
        for j in range(OUT):
            # spline: (N, IN)
            sp = (B_all * layer.spline_weight[j].unsqueeze(0)).sum(-1) * layer.scale_spline
            # base: (N, IN)
            bs = silu_all * layer.base_weight[j].unsqueeze(0) * layer.scale_base
            phi = (sp + bs).cpu().numpy()  # (N, IN)
            amp_mat[j] = phi.max(axis=0) - phi.min(axis=0)

    return amp_mat


# ════════════════════════════════════════════════════════════
# 2. 数据加载（可选，用于真实输入分布分析）
# ════════════════════════════════════════════════════════════
def load_hidden_features(h5_dir, label_csv, model, pos_feats, neg_feats,
                         device, max_slides=None):
    """
    从 h5 文件加载 patch features，经过 feat_proj 得到 hidden (N, 256)。
    返回 {'tumor': array(M,256), 'normal': array(K,256)}
    """
    import h5py, pandas as pd
    df = pd.read_csv(label_csv, encoding="utf-8-sig")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.columns = df.columns.str.strip()
    col_image = next((c for c in df.columns if c.lower() in ("image","filename","file","slide","name")), None)
    col_label = next((c for c in df.columns if c.lower() in ("type","label","class","category")), None)
    if col_image is None or col_label is None:
        raise KeyError(f"CSV 列名识别失败。实际列名: {df.columns.tolist()}")
    print(f"   CSV: image_col=\'{col_image}\'  label_col=\'{col_label}\'  ({len(df)} rows)")
    print("All labels:", set(df[col_label]))
    hidden_tumor, hidden_normal = [], []
    attn_tumor, attn_normal = [], []

    loaded = 0

    
    max_per_class = max_slides // 2 if max_slides else None

    count_tumor = 0
    count_normal = 0

    for i, row in df.iterrows():

        if max_per_class is not None:
            if count_tumor >= max_per_class and count_normal >= max_per_class:
                break

        stem = os.path.splitext(row[col_image])[0]
        h5 = os.path.join(h5_dir, f"{stem}.h5")
        if not os.path.exists(h5):
            continue
        with h5py.File(h5, "r") as f:
            feats = torch.from_numpy(f["features"][:]).float().to(device)

        feats = F.normalize(feats, dim=-1)
        with torch.no_grad():
            h_vec = model.feat_proj(feats).cpu().numpy()  # (N, 256)
            logits, attn_w, _ = model(feats, pos_feats, neg_feats)
            attn_np = attn_w.cpu().numpy()

        loaded += 1

        label = str(row[col_label]).strip().lower()

        if "tumor" in label:
            if max_per_class is None or count_tumor < max_per_class:
                hidden_tumor.append(h_vec)
                attn_tumor.append(attn_np)
                count_tumor += 1

        elif "normal" in label:
            if max_per_class is None or count_normal < max_per_class:
                hidden_normal.append(h_vec)
                attn_normal.append(attn_np)
                count_normal += 1

    result = {}
    if hidden_tumor:
        result["tumor"]  = np.concatenate(hidden_tumor,  axis=0)
        result["attn_tumor"]  = np.concatenate(attn_tumor,  axis=0)
    if hidden_normal:
        result["normal"] = np.concatenate(hidden_normal, axis=0)
        result["attn_normal"] = np.concatenate(attn_normal, axis=0)

    print("Tumor:", len(hidden_tumor))
    print("Normal:", len(hidden_normal))
    return result


# ════════════════════════════════════════════════════════════
# 3. 线性 MLP 等效拟合（R² 分析）
# ════════════════════════════════════════════════════════════
def compute_linear_r2(phi_vals: np.ndarray, x_vals: np.ndarray) -> float:
    """
    对 φ(x) 拟合一条线性回归 y = a*x + b，
    返回 R² —— 越低说明线性函数越无法覆盖该非线性激活。
    """
    slope, intercept, r, p, se = stats.linregress(x_vals, phi_vals)
    r2 = r ** 2
    return float(r2), slope, intercept


# ════════════════════════════════════════════════════════════
# 4. 绘图 Part 1 — Top-K B-spline 曲线可视化
# ════════════════════════════════════════════════════════════
DARK_BG  = "white"
GRID_COL = "#cccccc"
ACCENT   = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
            "#e07b00", "#0077bb", "#228b22", "#cc3311"]

def plot_top_spline_curves(layer: KANLinear, amp_mat: np.ndarray,
                            x_vals: np.ndarray, top_k: int,
                            output_path: str):
    """
    Part 1：绘制 Top-K 条边的 B-spline 激活曲线，附线性对比线和 R² 标注。
    """
    OUT, IN = amp_mat.shape
    flat_idx = np.argsort(amp_mat.ravel())[::-1][:top_k]
    top_pairs = [(int(idx // IN), int(idx % IN)) for idx in flat_idx]

    ncols = 4
    nrows = int(np.ceil(top_k / ncols))
    fig_w = ncols * 4.2
    fig_h = nrows * 3.5

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    fig.suptitle(
        f"KAN Layer 1  —  Top-{top_k} Edges by Activation Amplitude\n"
        f"(Blue = learned B-spline φ(x),  Dashed = best-fit linear,  "
        f"R² measures linearity — lower R² = stronger nonlinearity)",
        color="black", fontsize=13, fontweight="bold", y=1.01
    )

    for rank, (j, i) in enumerate(top_pairs):
        ax = fig.add_subplot(nrows, ncols, rank + 1)
        ax.set_facecolor(DARK_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.tick_params(colors="#444444", labelsize=7)
        ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle="--")

        phi = eval_single_edge(layer, i, j, x_vals)
        # 平滑曲线
        phi_smooth = gaussian_filter1d(phi, sigma=1.5)
        r2, slope, intercept = compute_linear_r2(phi_smooth, x_vals)
        linear_fit = slope * x_vals + intercept

        color = ACCENT[rank % len(ACCENT)]
        ax.plot(x_vals, phi_smooth, color=color, linewidth=2.0, label="φ(x) KAN")
        ax.plot(x_vals, linear_fit, color="black", linewidth=1.0,
                linestyle="--", alpha=0.55, label="Linear fit")
        ax.fill_between(x_vals, phi_smooth, linear_fit,
                        alpha=0.12, color=color)

        amp = amp_mat[j, i]
        ax.set_title(
            f"Edge  in[{i}] → hid[{j}]\n"
            f"Amplitude={amp:.4f}   R²={r2:.3f}",
            color="black", fontsize=10, pad=4
        )
        ax.set_xlabel("x (hidden input)", color="#444444", fontsize=10)
        ax.set_ylabel("φ(x)", color="#444444", fontsize=10)
        if rank == 0:
            ax.legend(fontsize=10, facecolor="#f5f5f5", labelcolor="black",
                      edgecolor="#aaaaaa", loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  ✅ Part 1 saved → {output_path}")


# ════════════════════════════════════════════════════════════
# 5. 绘图 Part 2 — Feature Attribution Heatmap
# ════════════════════════════════════════════════════════════
def plot_attribution_heatmap(amp_mat: np.ndarray, output_path: str,
                              top_k_labels: int = 20):
    """
    Part 2-A：振幅矩阵热力图（out_dim × in_dim），展示哪些连接最关键。
    """
    OUT, IN = amp_mat.shape

    # 聚合：每个 input dim 的最大振幅
    in_importance  = amp_mat.max(axis=0)   # (IN,)  → 哪个输入特征最被关注
    out_importance = amp_mat.max(axis=1)   # (OUT,) → 哪个隐层节点影响最大

    top_in  = np.argsort(in_importance)[::-1][:top_k_labels]
    top_out = np.argsort(out_importance)[::-1][:top_k_labels]

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")

    fig.suptitle(
        "KAN Feature Attribution — Amplitude Heatmap",
        color="black", fontsize=14, fontweight="bold"
    )

    ax.set_facecolor(DARK_BG)

    top_k_bar = 10
    top_in_bar = np.argsort(in_importance)[::-1][:top_k_bar]

    colors_bar = plt.cm.inferno(
        Normalize()(in_importance[top_in_bar])
    )

    ax.barh(
        range(top_k_bar),
        in_importance[top_in_bar],
        color=colors_bar,
        edgecolor="none"
    )

    ax.set_yticks(range(top_k_bar))
    ax.set_yticklabels(
        [f"dim {d}" for d in top_in_bar],
        fontsize=10,
        color="#444444"
    )

    ax.invert_yaxis()

    ax.set_title(
        f"Top-{top_k_bar} Input Dims by Max Amplitude",
        color="black",
        fontsize=11
    )

    ax.set_xlabel(
        "Max Amplitude across all output nodes",
        color="#444444"
    )

    ax.tick_params(colors="#444444")

    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)

    ax.grid(axis="x", color=GRID_COL, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"  ✅ Part 2-A saved → {output_path}")

    return top_in_bar

    # # 子矩阵：top input dims × top output dims
    # sub = amp_mat[np.ix_(top_out, top_in)]

    # fig, axes = plt.subplots(1, 1, figsize=(6, 6), facecolor="white")
    # fig.suptitle("KAN Feature Attribution — Amplitude Heatmap",
    #              color="black", fontsize=14, fontweight="bold")

    # # Panel A: 完整矩阵（降采样展示）
    # ax = axes[0]
    # ax.set_facecolor(DARK_BG)
    # step_out = max(1, OUT // 64)
    # step_in  = max(1, IN  // 64)
    # vis_mat  = amp_mat[::step_out, ::step_in]
    # im = ax.imshow(vis_mat, aspect="auto", cmap="inferno",
    #                interpolation="nearest")
    # ax.set_title(f"Full Amplitude Matrix\n({OUT}×{IN} edges, sampled)",
    #              color="black", fontsize=11)
    # ax.set_xlabel("Input dim (hidden features)", color="#444444")
    # ax.set_ylabel("Output dim (hidden → hidden/2)", color="#444444")
    # ax.tick_params(colors="#444444")
    # plt.colorbar(im, ax=ax, fraction=0.03).ax.yaxis.set_tick_params(color="black")

    # # Panel B: Top sub-matrix
    # ax = axes[1]
    # ax.set_facecolor(DARK_BG)
    # im2 = ax.imshow(sub, aspect="auto", cmap="inferno",
    #                 interpolation="nearest")
    # ax.set_xticks(range(len(top_in)))
    # ax.set_xticklabels(top_in, rotation=90, fontsize=10, color="#444444")
    # ax.set_yticks(range(len(top_out)))
    # ax.set_yticklabels(top_out, fontsize=10, color="#444444")
    # ax.set_title(f"Top-{top_k_labels} Inputs × Top-{top_k_labels} Outputs",
    #              color="black", fontsize=11)
    # ax.set_xlabel("Top input dims", color="#444444")
    # ax.set_ylabel("Top output dims", color="#444444")
    # plt.colorbar(im2, ax=ax, fraction=0.03).ax.yaxis.set_tick_params(color="black")

    # Panel C: Input importance bar chart
    # ax = axes[0]
    # ax.set_facecolor(DARK_BG)
    # top_k_bar = 10
    # top_in_bar = np.argsort(in_importance)[::-1][:top_k_bar]
    # colors_bar = plt.cm.inferno(
    #     Normalize()(in_importance[top_in_bar]))
    # ax.barh(range(top_k_bar), in_importance[top_in_bar],
    #         color=colors_bar, edgecolor="none")
    # ax.set_yticks(range(top_k_bar))
    # ax.set_yticklabels([f"dim {d}" for d in top_in_bar],
    #                    fontsize=10, color="#444444")
    # ax.invert_yaxis()
    # ax.set_title(f"Top-{top_k_bar} Input Dims by Max Amplitude",
    #              color="black", fontsize=11)
    # ax.set_xlabel("Max Amplitude across all output nodes", color="#444444")
    # ax.tick_params(colors="#444444")
    # for spine in ax.spines.values():
    #     spine.set_edgecolor(GRID_COL)
    # ax.grid(axis="x", color=GRID_COL, linewidth=0.5)

    # plt.tight_layout()
    # plt.savefig(output_path, dpi=150, bbox_inches="tight",
    #             facecolor="white")
    # plt.close()
    # print(f"  ✅ Part 2-A saved → {output_path}")

    # return top_in[:10]   # 返回前10个最重要 input dim，供后续使用



# ════════════════════════════════════════════════════════════
# 6. 绘图 Part 2-B — 真实输入分布下的响应分析（需数据）
# ════════════════════════════════════════════════════════════
def plot_real_distribution_response(layer: KANLinear,
                                     hidden_data: dict,
                                     amp_mat: np.ndarray,
                                     top_dims: list,
                                     output_path: str):
    """
    Part 2-B：对 Top-K 个输入 dim，分别绘制：
      · 肿瘤 vs 正常 patch 的输入分布（KDE）
      · 对应的 φ(x) 激活曲线
    → 直观展示 KAN 如何在肿瘤特征区间产生不同响应。
    """
    has_tumor  = "tumor"  in hidden_data
    has_normal = "normal" in hidden_data

    if not (has_tumor or has_normal):
        print("  ⚠️  无真实数据，跳过 Part 2-B")
        return

    n_dims = len(top_dims)
    fig, axes = plt.subplots(2, n_dims, figsize=(n_dims * 3.5, 7),
                              facecolor="white")
    if n_dims == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        "KAN Activation Response vs Real Input Distribution\n"
        "(Top input dims by amplitude — tumor vs normal patch features)",
        color="black", fontsize=20, fontweight="bold"
    )

    x_grid = np.linspace(-1.5, 1.5, 300)

    for col, dim_idx in enumerate(top_dims):
        # 找该 dim 贡献最大的 out_idx
        best_out = int(np.argmax(amp_mat[:, dim_idx]))
        phi = eval_single_edge(layer, dim_idx, best_out, x_grid)
        phi_smooth = gaussian_filter1d(phi, sigma=2.0)

        # Row 0: KDE of actual inputs
        ax_kde = axes[0, col]
        ax_kde.set_facecolor(DARK_BG)
        for spine in ax_kde.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax_kde.tick_params(colors="#444444", labelsize=6)
        ax_kde.grid(True, color=GRID_COL, linewidth=0.4, linestyle="--")

        if has_tumor:
            vals_t = hidden_data["tumor"][:, dim_idx]
            kde_t  = stats.gaussian_kde(vals_t, bw_method=0.3)
            kde_x  = np.linspace(vals_t.min(), vals_t.max(), 200)
            ax_kde.fill_between(kde_x, kde_t(kde_x), alpha=0.4,
                                color="#f78166", label="Tumor")
            ax_kde.plot(kde_x, kde_t(kde_x), color="#f78166", linewidth=1.5)

        if has_normal:
            vals_n = hidden_data["normal"][:, dim_idx]
            kde_n  = stats.gaussian_kde(vals_n, bw_method=0.3)
            kde_x  = np.linspace(vals_n.min(), vals_n.max(), 200)
            ax_kde.fill_between(kde_x, kde_n(kde_x), alpha=0.4,
                                color="#58a6ff", label="Normal")
            ax_kde.plot(kde_x, kde_n(kde_x), color="#58a6ff", linewidth=1.5)

        ax_kde.set_title(f"dim {dim_idx}\n→ hid[{best_out}]",
                         color="black", fontsize=15)
        ax_kde.set_ylabel("Density", color="#444444", fontsize=15)
        if col == 0:
            ax_kde.legend(fontsize=15, facecolor="#f5f5f5",
                          labelcolor="black", edgecolor="#aaaaaa")

        # Row 1: φ(x) activation curve
        ax_phi = axes[1, col]
        ax_phi.set_facecolor(DARK_BG)
        for spine in ax_phi.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax_phi.tick_params(colors="#444444", labelsize=10)
        ax_phi.grid(True, color=GRID_COL, linewidth=0.4, linestyle="--")

        color_phi = ACCENT[col % len(ACCENT)]
        ax_phi.plot(x_grid, phi_smooth, color=color_phi, linewidth=2.0)
        ax_phi.fill_between(x_grid, phi_smooth,
                            alpha=0.15, color=color_phi)

        # 线性拟合
        r2, slope, intercept = compute_linear_r2(phi_smooth, x_grid)
        ax_phi.plot(x_grid, slope * x_grid + intercept,
                    "--", color="#666666", linewidth=1.0, alpha=0.5)

        # 标注肿瘤 / 正常区间
        if has_tumor:
            m_t = hidden_data["tumor"][:, dim_idx].mean()
            ax_phi.axvline(m_t, color="#f78166", linewidth=1.2,
                           linestyle=":", alpha=0.85)
        if has_normal:
            m_n = hidden_data["normal"][:, dim_idx].mean()
            ax_phi.axvline(m_n, color="#58a6ff", linewidth=1.2,
                           linestyle=":", alpha=0.85)

        amp = amp_mat[:, dim_idx].max()
        ax_phi.set_title(f"φ(x)   Amp={amp:.4f}   R²={r2:.3f}",
                         color="black", fontsize=15)
        ax_phi.set_xlabel("x (feature value)", color="#444444", fontsize=15)
        ax_phi.set_ylabel("φ(x)", color="#444444", fontsize=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  ✅ Part 2-B saved → {output_path}")


# ════════════════════════════════════════════════════════════
# # 7. 绘图 Part 3 — MLP vs KAN 非线性程度量化
# # ════════════════════════════════════════════════════════════
# def plot_nonlinearity_evidence(layer: KANLinear, amp_mat: np.ndarray,
#                                 x_vals: np.ndarray,
#                                 top_k: int, output_path: str):
#     """
#     Part 3：计算所有 Top-K 边的 R² 分布。
#     R² 分布越偏向 0，说明 KAN 学到的函数越无法被线性覆盖。
#     同时画出一个"假设 MLP"的理论 R²=1.0 参考线做对比。
#     """
#     OUT, IN = amp_mat.shape
#     flat_idx = np.argsort(amp_mat.ravel())[::-1][:top_k]
#     top_pairs = [(int(idx // IN), int(idx % IN)) for idx in flat_idx]

#     r2_list = []
#     amp_list = []
#     for j, i in top_pairs:
#         phi = eval_single_edge(layer, i, j, x_vals)
#         phi_s = gaussian_filter1d(phi, sigma=1.5)
#         r2, _, _ = compute_linear_r2(phi_s, x_vals)
#         r2_list.append(r2)
#         amp_list.append(amp_mat[j, i])

#     r2_arr  = np.array(r2_list)
#     amp_arr = np.array(amp_list)

#     fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="white")
#     fig.suptitle(
#         "MLP vs KAN — Nonlinearity Quantification\n"
#         "(R² of best-fit linear function: lower = more nonlinear, "
#         "stronger evidence for KAN)",
#         color="black", fontsize=13, fontweight="bold"
#     )

#     # Panel A: R² histogram
#     ax = axes[0]
#     ax.set_facecolor(DARK_BG)
#     for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)
#     ax.tick_params(colors="#444444")
#     ax.grid(axis="y", color=GRID_COL, linewidth=0.5)
#     n_bins = min(30, top_k // 3 + 5)
#     ax.hist(r2_arr, bins=n_bins, color="#58a6ff", edgecolor="white",
#             alpha=0.85, label=f"KAN edges (n={top_k})")
#     ax.axvline(1.0, color="#f78166", linewidth=2.0, linestyle="--",
#                label="MLP theoretical R²=1")
#     ax.axvline(r2_arr.mean(), color="#3fb950", linewidth=1.5, linestyle=":",
#                label=f"KAN mean R²={r2_arr.mean():.3f}")
#     ax.set_xlabel("R² (linear fit)", color="#444444")
#     ax.set_ylabel("Count", color="#444444")
#     ax.set_title(f"R² Distribution of Top-{top_k} Edges", color="black", fontsize=11)
#     ax.legend(fontsize=10, facecolor="#f5f5f5", labelcolor="black",
#               edgecolor="#aaaaaa")

#     # Panel B: R² vs Amplitude scatter
#     ax = axes[1]
#     ax.set_facecolor(DARK_BG)
#     for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)
#     ax.tick_params(colors="#444444")
#     ax.grid(color=GRID_COL, linewidth=0.4)
#     sc = ax.scatter(amp_arr, r2_arr, c=r2_arr, cmap="RdYlGn_r",
#                     s=40, alpha=0.75, edgecolors="none")
#     plt.colorbar(sc, ax=ax, fraction=0.03, label="R²"
#                  ).ax.yaxis.set_tick_params(color="black")
#     ax.set_xlabel("Amplitude", color="#444444")
#     ax.set_ylabel("R² (linearity)", color="#444444")
#     ax.set_title("Amplitude vs R²\n(top-right = high amplitude & linear;\n"
#                  "top-left = best KAN advantage zone)", color="black", fontsize=10)
#     # Mark the "KAN wins" zone
#     ax.axhline(0.5, color="#ffa657", linewidth=1.0, linestyle="--", alpha=0.6)
#     ax.text(amp_arr.max() * 0.05, 0.45,
#             "← Highly nonlinear edges (KAN exclusive)",
#             color="#ffa657", fontsize=10)

#     # Panel C: cumulative R² percentile
#     ax = axes[2]
#     ax.set_facecolor(DARK_BG)
#     for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)
#     ax.tick_params(colors="#444444")
#     ax.grid(color=GRID_COL, linewidth=0.4)
#     sorted_r2 = np.sort(r2_arr)
#     cdf = np.arange(1, len(sorted_r2) + 1) / len(sorted_r2)
#     ax.plot(sorted_r2, cdf, color="#58a6ff", linewidth=2.0)
#     ax.fill_betweenx(cdf, sorted_r2, 1.0, alpha=0.12, color="#f78166",
#                      label="MLP can't cover (R² < threshold)")
#     ax.axvline(0.9, color="#ffa657", linewidth=1.2, linestyle="--")
#     pct_below_09 = (r2_arr < 0.9).mean() * 100
#     ax.text(0.92, 0.05, f"{pct_below_09:.1f}% edges\nhave R²<0.9",
#             color="#ffa657", fontsize=10,
#             bbox=dict(facecolor="#f5f5f5", alpha=0.7, edgecolor="none"))
#     ax.set_xlabel("R² (linear fit)", color="#444444")
#     ax.set_ylabel("CDF", color="#444444")
#     ax.set_title("Cumulative R² Distribution\n(CDF — area right of MLP threshold = KAN gain)",
#                  color="black", fontsize=10)
#     ax.legend(fontsize=10, facecolor="#f5f5f5", labelcolor="black",
#               edgecolor="#aaaaaa")

#     plt.tight_layout()
#     plt.savefig(output_path, dpi=150, bbox_inches="tight",
#                 facecolor="white")
#     plt.close()
#     print(f"  ✅ Part 3 saved → {output_path}")

#     # 打印汇总统计
#     print(f"\n  📊 Nonlinearity Summary (Top-{top_k} edges):")
#     print(f"     R² mean  = {r2_arr.mean():.4f}")
#     print(f"     R² median= {np.median(r2_arr):.4f}")
#     print(f"     R² < 0.9 : {pct_below_09:.1f}%  edges can't be linearized")
#     print(f"     R² < 0.5 : {(r2_arr < 0.5).mean()*100:.1f}%  strongly nonlinear")


# ════════════════════════════════════════════════════════════
# 8. 绘图 Part 4 — Layer 2 输出权重分析（128→2 分类头）
# ════════════════════════════════════════════════════════════
def plot_classification_head(layer2: KANLinear, amp_mat1: np.ndarray,
                              output_path: str):
    """
    Part 4：可视化 KAN 第二层（128→2）的激活函数，
    分别看 tumor logit (out=1) 和 normal logit (out=0) 的输入贡献。
    """
    if layer2 is None:
        return

    x_vals = np.linspace(-1.0, 1.0, 300)
    IN2 = layer2.in_features   # 128
    amp_tumor  = np.zeros(IN2)
    amp_normal = np.zeros(IN2)

    # 评估所有 128 个输入维度对 tumor (out=1) 和 normal (out=0) 的贡献
    with torch.no_grad():
        device = layer2.base_weight.device
        x_t = torch.tensor(x_vals, dtype=torch.float32, device=device)
        x_full = torch.zeros(len(x_vals), IN2, device=device)
        for i in range(IN2):
            x_full[:, i] = x_t

        B_all = layer2.b_splines(x_full)    # (N, 128, n_basis)
        silu  = F.silu(x_t)

        for i in range(IN2):
            # tumor (out=1)
            sp1 = (B_all[:, i, :] * layer2.spline_weight[1, i, :].unsqueeze(0)
                   ).sum(1) * layer2.scale_spline
            bs1 = silu * layer2.base_weight[1, i] * layer2.scale_base
            phi1 = (sp1 + bs1).cpu().numpy()
            amp_tumor[i] = phi1.max() - phi1.min()

            # normal (out=0)
            sp0 = (B_all[:, i, :] * layer2.spline_weight[0, i, :].unsqueeze(0)
                   ).sum(1) * layer2.scale_spline
            bs0 = silu * layer2.base_weight[0, i] * layer2.scale_base
            phi0 = (sp0 + bs0).cpu().numpy()
            amp_normal[i] = phi0.max() - phi0.min()

    diff = amp_tumor - amp_normal   # positive = more tumor-specific

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="white")
    fig.suptitle(
        "KAN Layer 2 (128→2) — Classification Head Analysis\n"
        "Which hidden features drive Tumor vs Normal prediction?",
        color="black", fontsize=13, fontweight="bold"
    )

    for ax in axes:
        ax.set_facecolor(DARK_BG)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)
        ax.tick_params(colors="#444444")

    # # Panel A: Tumor amplitude
    # top10t = np.argsort(amp_tumor)[::-1][:20]
    # axes[0].barh(range(20), amp_tumor[top10t],
    #              color="#f78166", edgecolor="none", alpha=0.85)
    # axes[0].set_yticks(range(20))
    # axes[0].set_yticklabels([f"hid[{d}]" for d in top10t],
    #                          fontsize=10, color="#444444")
    # axes[0].invert_yaxis()
    # axes[0].set_title("Top-20 → Tumor logit contribution", color="black", fontsize=10)
    # axes[0].set_xlabel("Amplitude", color="#444444")
    # axes[0].grid(axis="x", color=GRID_COL, linewidth=0.4)

    # Panel B: Differential (tumor - normal)
    top20d = np.argsort(np.abs(diff))[::-1][:10]
    colors_d = ["#f78166" if diff[d] > 0 else "#58a6ff" for d in top20d]
    axes[1].barh(range(10), diff[top20d], color=colors_d,
                 edgecolor="none", alpha=0.85)
    axes[1].set_yticks(range(10))
    axes[1].set_yticklabels([f"hid[{d}]" for d in top20d],
                             fontsize=10, color="#444444")
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#555555", linewidth=0.8, alpha=0.4)
    axes[1].set_title("Differential (Tumor − Normal amplitude)\nRed=tumor specific, Blue=normal specific",
                       color="black", fontsize=10)
    axes[1].set_xlabel("Δ Amplitude", color="#444444")
    axes[1].grid(axis="x", color=GRID_COL, linewidth=0.4)

    # Panel C: 画出最具鉴别力的 top-4 边的激活曲线
    top4_diff = np.argsort(np.abs(diff))[::-1][:4]
    colors_p = ["#f78166", "#58a6ff", "#3fb950", "#d2a8ff"]
    for k, dim_i in enumerate(top4_diff):
        out_tumor = 1
        phi_t = eval_single_edge(layer2, dim_i, out_tumor, x_vals)
        phi_n = eval_single_edge(layer2, dim_i, 0, x_vals)
        phi_t_s = gaussian_filter1d(phi_t, sigma=1.5)
        phi_n_s = gaussian_filter1d(phi_n, sigma=1.5)
        c = colors_p[k]
        axes[2].plot(x_vals, phi_t_s, color=c, linewidth=1.8,
                     label=f"hid[{dim_i}]→tumor")
        axes[2].plot(x_vals, phi_n_s, color=c, linewidth=1.0,
                     linestyle="--", alpha=0.5)
    axes[2].set_title("Top-4 differential edge φ(x)\n(solid=→tumor, dashed=→normal)",
                       color="black", fontsize=10)
    axes[2].set_xlabel("x (hidden feature value)", color="#444444")
    axes[2].set_ylabel("φ(x)", color="#444444")
    axes[2].legend(fontsize=10, facecolor="#f5f5f5",
                   labelcolor="black", edgecolor="#aaaaaa")
    axes[2].grid(color=GRID_COL, linewidth=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  ✅ Part 4 saved → {output_path}")


# ════════════════════════════════════════════════════════════
# 9. Main
# ════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="KAN Interpretability Analysis for TextGuidedMIL_KAN"
    )
    # Required
    p.add_argument("--ckpt",          required=True, help="模型 checkpoint (.pt)")
    p.add_argument("--output",        default="./kan_analysis", help="输出目录")

    # Optional (for real-data analysis)
    p.add_argument("--feat_dir",      default="", help="patch feature .h5 文件目录")
    p.add_argument("--h5_dir",        default="", help="同 --feat_dir（兼容旧参数名）")
    p.add_argument("--label_csv",     default="", help="标签 CSV")
    p.add_argument("--text_feat_dir", default="", help="pos/neg_features.npy 目录")
    p.add_argument("--max_slides",    type=int, default=200,
                   help="最多加载多少张 slide（加快速度）")

    # Model config
    p.add_argument("--in_dim",      type=int,   default=512)
    p.add_argument("--hidden_dim",  type=int,   default=256)
    p.add_argument("--kan_grid",    type=int,   default=6)
    p.add_argument("--dropout",     type=float, default=0.25)
    p.add_argument("--alpha_init",  type=float, default=0.5)
    p.add_argument("--use_neg",     type=lambda x: x.lower()=="true", default=True)
    p.add_argument("--neg_weight",  type=float, default=0.5)

    # Analysis config
    p.add_argument("--top_k_curves",    type=int, default=2,
                   help="Part 1：展示几条 B-spline 曲线")
    p.add_argument("--top_k_nonlinear", type=int, default=200,
                   help="Part 3：对多少条边做 R² 分析")
    p.add_argument("--top_k_response",  type=int, default=2,
                   help="Part 2-B：展示几个输入 dim 的响应")
    p.add_argument("--x_range",         type=float, default=1.0,
                   help="B-spline 评估的输入范围 [-x_range, x_range]")
    p.add_argument("--n_x_pts",         type=int,   default=500,
                   help="B-spline 评估点数")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output: {args.output}")

    # ── 加载模型 ─────────────────────────────────────────────
    print(f"\n🔄 Loading checkpoint: {args.ckpt}")
    model = TextGuidedMIL_KAN(
        in_dim=args.in_dim, hidden_dim=args.hidden_dim,
        n_classes=2, dropout=args.dropout,
        alpha_init=args.alpha_init, use_neg=args.use_neg,
        neg_weight=args.neg_weight, kan_grid=args.kan_grid,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f"   Learned alpha = {model.alpha.item():.4f}")

    layer1 = get_kan_layer(model)
    layer2 = get_kan_layer2(model)
    print(f"   KAN Layer 1: {layer1.in_features} → {layer1.out_features}  "
          f"(grid={layer1.grid_size}, order={layer1.spline_order})")
    if layer2:
        print(f"   KAN Layer 2: {layer2.in_features} → {layer2.out_features}")

    # 评估用的 x 轴（覆盖 B-spline 的 grid_range 并略微扩展）
    x_vals = np.linspace(-args.x_range, args.x_range, args.n_x_pts)

    # ── 计算振幅矩阵 ─────────────────────────────────────────
    print(f"\n⚙️  Computing amplitude matrix ({layer1.out_features}×{layer1.in_features})...")
    print(f"   This may take a minute for large layers...")
    amp_mat = compute_amplitude_matrix(layer1, x_vals)
    print(f"   Done. Amplitude range: [{amp_mat.min():.5f}, {amp_mat.max():.5f}]")
    print(f"   Mean amplitude: {amp_mat.mean():.5f}")

    # ── Part 1: B-spline 曲线可视化 ──────────────────────────
    print(f"\n📈 Part 1: B-spline curve visualization (Top-{args.top_k_curves} edges)...")
    plot_top_spline_curves(
        layer1, amp_mat, x_vals,
        top_k=args.top_k_curves,
        output_path=os.path.join(args.output, "part1_bspline_curves.svg")
    )

    # ── Part 2-A: Attribution Heatmap ────────────────────────
    print(f"\n🔥 Part 2-A: Feature attribution heatmap...")
    top_important_dims = plot_attribution_heatmap(
        amp_mat,
        output_path=os.path.join(args.output, "part2a_attribution_heatmap.svg"),
        top_k_labels=20
    )

    # ── Part 2-B: Real distribution response（需数据）────────
    # 兼容 --feat_dir 和 --h5_dir 两种参数名
    effective_h5_dir = args.feat_dir or args.h5_dir
    has_data = (effective_h5_dir and args.label_csv and args.text_feat_dir
                and os.path.exists(effective_h5_dir) and os.path.exists(args.label_csv))

    if has_data:
        print(f"\n📂 Part 2-B: Loading real features for distribution analysis...")
        print(f"   feat_dir: {effective_h5_dir}")
        pos_feats = torch.from_numpy(
            np.load(os.path.join(args.text_feat_dir, "pos_features.npy")).astype(np.float32)
        ).to(device)
        neg_path = os.path.join(args.text_feat_dir, "neg_features.npy")
        neg_feats = (torch.from_numpy(np.load(neg_path).astype(np.float32)).to(device)
                     if os.path.exists(neg_path) else None)

        hidden_data = load_hidden_features(
            effective_h5_dir, args.label_csv, model,
            pos_feats, neg_feats, device,
            max_slides=args.max_slides
        )
        for k, v in hidden_data.items():
            if isinstance(v, np.ndarray):
                print(f"   {k}: {v.shape}")

        top_dims_response = list(top_important_dims[:args.top_k_response])
        plot_real_distribution_response(
            layer1, hidden_data, amp_mat,
            top_dims=top_dims_response,
            output_path=os.path.join(args.output, "part2b_real_response.svg")
        )
    else:
        print(f"\n  ⚠️  Part 2-B skipped (no --h5_dir / --label_csv / --text_feat_dir provided)")

    # ── Part 3: MLP vs KAN 非线性量化 ────────────────────────
    # print(f"\n📊 Part 3: Nonlinearity quantification (Top-{args.top_k_nonlinear} edges)...")
    # plot_nonlinearity_evidence(
    #     layer1, amp_mat, x_vals,
    #     top_k=args.top_k_nonlinear,
    #     output_path=os.path.join(args.output, "part3_mlp_vs_kan_nonlinearity.svg")
    # )

    # ── Part 4: Classification head analysis ─────────────────
    if layer2 is not None:
        print(f"\n🎯 Part 4: Classification head analysis (Layer 2)...")
        plot_classification_head(
            layer2, amp_mat,
            output_path=os.path.join(args.output, "part4_classification_head.svg")
        )

    # ── 保存振幅矩阵（供后续分析）────────────────────────────
    amp_save = os.path.join(args.output, "amplitude_matrix.npy")
    np.save(amp_save, amp_mat)
    print(f"\n  💾 Amplitude matrix saved → {amp_save}")

    # ── 汇总报告 ─────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  KAN Interpretability Analysis Complete")
    print(f"{'═'*60}")
    print(f"  Output files:")
    for fname in sorted(os.listdir(args.output)):
        fpath = os.path.join(args.output, fname)
        size  = os.path.getsize(fpath) / 1024
        print(f"    {fname}  ({size:.0f} KB)")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()