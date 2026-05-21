"""
text_guided_mil_kan.py
----------------------
TextGuidedMIL with KAN (Kolmogorov-Arnold Network) classifier.

与 text_guided_mil.py 完全相同，唯一区别是把最后的 MLP classifier
替换为 KAN classifier。其余结构（CPEG、gated attention、text guidance）保持不变。

KAN 实现：
  - 每个连接都是一个可学习的 1D 函数（B-spline + residual base）
  - φ(x) = w_b * SiLU(x) + w_s * Σ c_k * B_k(x)
  - 参数量略大于 MLP，但表达能力更强（对低维非线性更敏感）

使用方式和 text_guided_mil.py 完全一样，直接替换 import 即可。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════
# KAN Layer
# ════════════════════════════════════════════════════════════
class KANLinear(nn.Module):
    """
    一层 KAN：将 in_features 映射到 out_features。
    每对 (i, j) 之间有一个可学习的 B-spline 函数。

    Args:
        in_features  : 输入维度
        out_features : 输出维度
        grid_size    : B-spline 控制点数（越大越精细，默认5已够用）
        spline_order : B-spline 阶数（3 = cubic，推荐）
        scale_noise  : 初始化时 spline 权重的噪声幅度
        scale_base   : base (SiLU) 分支的初始权重
        scale_spline : spline 分支的初始权重
        grid_range   : 输入值的预期范围（用于归一化 grid）
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        grid_range: tuple = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order

        # B-spline grid: (in_features, grid_size + 2*spline_order + 1)
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float)
            * h + grid_range[0]
        )  # (grid_size + 2*spline_order + 1,)
        grid = grid.unsqueeze(0).expand(in_features, -1)  # (in, G)
        self.register_buffer("grid", grid)

        # 可学习参数
        # base 权重（SiLU residual 分支）
        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        # spline 系数: (out, in, n_basis)
        n_basis = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, n_basis)
        )
        self.scale_base   = scale_base
        self.scale_spline = scale_spline

        self._init_weights(scale_noise)

    def _init_weights(self, scale_noise):
        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)
        with torch.no_grad():
            noise = (
                torch.rand(self.spline_weight.shape) * 2 - 1
            ) * scale_noise
            self.spline_weight.copy_(noise)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 B-spline 基函数值。
        x    : (..., in_features)
        返回 : (..., in_features, n_basis)
        """
        *batch, D = x.shape
        x = x.reshape(-1, D)  # (B, in)
        x = x.unsqueeze(-1)   # (B, in, 1)
        grid = self.grid       # (in, G)

        # 递推计算 B-spline（Cox-de Boor）
        # 初始化：阶0（指示函数）
        basis = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()
        # (B, in, G-1)

        for k in range(1, self.spline_order + 1):
            # 左分量
            d_left  = grid[:, k:-1]   - grid[:, :-(k+1)]   # (in, G-k-1)
            d_right = grid[:, k+1:]   - grid[:, 1:-k]      # (in, G-k-1)

            left  = (x - grid[:, :-(k+1)]) / (d_left   + 1e-8) * basis[:, :, :-1]
            right = (grid[:, k+1:] - x)    / (d_right  + 1e-8) * basis[:, :,  1:]
            basis = left + right  # (B, in, G-k)

        basis = basis.reshape(*batch, D, -1)  # (..., in, n_basis)
        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features)  →  (..., out_features)
        """
        # Base 分支：SiLU linear
        base_out = F.linear(F.silu(x), self.base_weight * self.scale_base)
        # (..., out)

        # Spline 分支
        B = self.b_splines(x)                      # (..., in, n_basis)
        *batch, D, nb = B.shape
        B_flat = B.reshape(-1, D * nb)             # (prod(batch), in*nb)
        W_flat = self.spline_weight.reshape(
            self.out_features, D * nb
        ) * self.scale_spline                       # (out, in*nb)
        spline_out = F.linear(B_flat, W_flat)       # (prod(batch), out)
        spline_out = spline_out.reshape(*batch, self.out_features)

        return base_out + spline_out


class KAN(nn.Module):
    """
    多层 KAN，用于替换 MLP classifier。

    Args:
        dims       : 各层维度列表，如 [256, 64, 2]
        grid_size  : B-spline 控制点数
        dropout    : 层间 dropout
    """
    def __init__(self, dims: list, grid_size: int = 5, dropout: float = 0.25):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(KANLinear(dims[i], dims[i + 1], grid_size=grid_size))
            if i < len(dims) - 2:           # 最后一层不加 dropout/LayerNorm
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ════════════════════════════════════════════════════════════
# TextGuidedMIL with KAN Classifier
# ════════════════════════════════════════════════════════════
class TextGuidedMIL_KAN(nn.Module):
    """
    TextGuidedMIL，将 MLP classifier 替换为 KAN classifier。

    与原版 TextGuidedMIL 接口完全相同：
      forward(patch_feats, pos_feats, neg_feats=None)
      → logits (1, n_classes), attn_raw (N,), sim_score (N,)

    Args:
        in_dim      : patch feature dimension (512 for CONCH)
        hidden_dim  : attention hidden dimension
        n_classes   : 输出类别数
        dropout     : dropout rate
        alpha_init  : text guidance 强度初始值（可学习）
        use_neg     : 是否使用 negative queries 对比引导
        neg_weight  : negative similarity 的减权重
        kan_grid    : KAN B-spline 控制点数（5~10，越大越精细）
    """
    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 256,
        n_classes: int = 2,
        dropout: float = 0.25,
        alpha_init: float = 0.5,
        use_neg: bool = False,
        neg_weight: float = 0.5,
        kan_grid: int = 5,
    ):
        super().__init__()
        self.use_neg    = use_neg
        self.neg_weight = neg_weight

        # 可学习 alpha：控制 text guidance 强度
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # Feature projection（与原版相同）
        self.feat_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gated Attention（与原版相同）
        self.attention_V = nn.Linear(hidden_dim, 128)
        self.attention_U = nn.Linear(hidden_dim, 128)
        self.attention_W = nn.Linear(128, 1)

        # Text similarity projection（与原版相同）
        self.sim_proj = nn.Linear(1, 1, bias=False)

        # ── KAN Classifier（替换原版 MLP）──────────────────────
        self.classifier = KAN(
            dims      = [hidden_dim, hidden_dim // 2, n_classes],
            grid_size = kan_grid,
            dropout   = dropout,
        )

    def compute_text_sim(self, patch_feats, pos_feats, neg_feats=None):
        """
        patch_feats : (N, 512) L2归一化
        pos_feats   : (P, 512) L2归一化
        neg_feats   : (Q, 512) or None
        返回        : (N, 1)
        """
        pos_sim = (patch_feats @ pos_feats.T).mean(dim=1, keepdim=True)

        if self.use_neg and neg_feats is not None:
            neg_sim = (patch_feats @ neg_feats.T).mean(dim=1, keepdim=True)
            return pos_sim - self.neg_weight * neg_sim

        return pos_sim

    def forward(self, patch_feats, pos_feats, neg_feats=None):
        """
        Args:
            patch_feats : (N, 512)
            pos_feats   : (P, 512)
            neg_feats   : (Q, 512) or None
        Returns:
            logits      : (1, n_classes)
            attn_raw    : (N,)
            sim_score   : (N,)
        """
        # Step 1: text similarity
        sim_score = self.compute_text_sim(patch_feats, pos_feats, neg_feats)  # (N,1)

        # Step 2: feature projection
        h = self.feat_proj(patch_feats)  # (N, hidden)

        # Step 3: gated attention + text guidance
        A_V = torch.tanh(self.attention_V(h))
        A_U = torch.sigmoid(self.attention_U(h))
        A   = self.attention_W(A_V * A_U)                      # (N,1)

        text_bias    = self.alpha * self.sim_proj(sim_score)   # (N,1)
        A_guided     = A + text_bias                           # (N,1)
        attn_weights = F.softmax(A_guided, dim=0)              # (N,1)

        # Step 4: aggregation
        bag = (attn_weights * h).sum(dim=0, keepdim=True)      # (1, hidden)

        # Step 5: KAN classification
        logits = self.classifier(bag)                          # (1, n_classes)

        return logits, A_guided.squeeze(), sim_score.squeeze()


def build_text_guided_mil_kan(
    in_dim: int = 512,
    hidden_dim: int = 256,
    n_classes: int = 2,
    dropout: float = 0.25,
    alpha_init: float = 0.5,
    use_neg: bool = False,
    neg_weight: float = 0.5,
    kan_grid: int = 5,
) -> TextGuidedMIL_KAN:
    return TextGuidedMIL_KAN(
        in_dim     = in_dim,
        hidden_dim = hidden_dim,
        n_classes  = n_classes,
        dropout    = dropout,
        alpha_init = alpha_init,
        use_neg    = use_neg,
        neg_weight = neg_weight,
        kan_grid   = kan_grid,
    )