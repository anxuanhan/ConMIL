
import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════
# KAN Layer
# ════════════════════════════════════════════════════════════
class KANLinear(nn.Module):
 
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

        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

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
 
        *batch, D = x.shape
        x = x.reshape(-1, D)  # (B, in)
        x = x.unsqueeze(-1)   # (B, in, 1)
        grid = self.grid       # (in, G)

   
        basis = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()
        # (B, in, G-1)

        for k in range(1, self.spline_order + 1):
           
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
   
        base_out = F.linear(F.silu(x), self.base_weight * self.scale_base)
        # (..., out)

   
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

    def __init__(self, dims: list, grid_size: int = 5, dropout: float = 0.25):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(KANLinear(dims[i], dims[i + 1], grid_size=grid_size))
            if i < len(dims) - 2:          
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ════════════════════════════════════════════════════════════
# TextGuidedMIL with KAN Classifier
# ════════════════════════════════════════════════════════════
class TextGuidedMIL_KAN(nn.Module):
  
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

        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        self.feat_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.attention_V = nn.Linear(hidden_dim, 128)
        self.attention_U = nn.Linear(hidden_dim, 128)
        self.attention_W = nn.Linear(128, 1)

   
        self.sim_proj = nn.Linear(1, 1, bias=False)

        self.classifier = KAN(
            dims      = [hidden_dim, hidden_dim // 2, n_classes],
            grid_size = kan_grid,
            dropout   = dropout,
        )

    def compute_text_sim(self, patch_feats, pos_feats, neg_feats=None):

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