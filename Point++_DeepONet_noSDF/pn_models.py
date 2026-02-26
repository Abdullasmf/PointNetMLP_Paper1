from typing import List, Tuple, Optional, Dict, Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PointNetMLPJoint", "PointNet2Encoder2D", "SetAbstraction", "MLP"]


def farthest_point_sampling(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
    # Farthest Point Sampling (FPS) indices
    device = xyz.device
    B, N, _ = xyz.shape
    n_samples = min(n_samples, N)
    idx = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    distances = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, device=device)
    for i in range(n_samples):
        idx[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # [B,1,2]
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # [B,N]
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = torch.max(distances, dim=1).indices
    return idx


def ball_query(
    xyz: torch.Tensor, centers: torch.Tensor, radius: float, max_k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Naive radius (ball) query capped at max_k with fallback to KNN when no neighbors
    _, N, _ = xyz.shape
    d2 = torch.cdist(centers, xyz, p=2) ** 2  # [B,M,N]
    within = d2 <= (radius**2)
    d2_masked = d2.clone()
    d2_masked[~within] = float("inf")
    k_init = min(max_k, N)
    idx = torch.topk(-d2_masked, k=k_init, dim=-1).indices
    gathered = torch.gather(d2_masked, dim=2, index=idx)
    mask = torch.isfinite(gathered)
    zero_mask = mask.sum(dim=-1) == 0  # [B,M]
    if zero_mask.any():
        idx_knn = torch.topk(-d2, k=k_init, dim=-1).indices
        zm_exp = zero_mask.unsqueeze(-1).expand_as(idx)
        idx = torch.where(zm_exp, idx_knn, idx)
        mask = torch.where(zm_exp, torch.ones_like(mask, dtype=torch.bool), mask)
    if idx.shape[-1] < max_k:
        pad = max_k - idx.shape[-1]
        idx = F.pad(idx, (0, pad), value=0)
        mask = F.pad(mask, (0, pad), value=False)
    elif idx.shape[-1] > max_k:
        idx = idx[..., :max_k]
        mask = mask[..., :max_k]
    return idx, mask


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: List[int],
        out_dim: int,
        act=nn.ReLU,
        norm: str = "batch",  # 'batch' | 'layer' | 'group' | 'none'
        num_groups: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = [in_dim] + hidden
        layers: List[nn.Module] = []

        def _norm(ch: int) -> Optional[nn.Module]:
            if norm == "batch":
                return nn.BatchNorm1d(ch)
            if norm == "layer":
                return nn.LayerNorm(ch)
            if norm == "group":
                g = max(1, min(num_groups, ch))
                # ensure divisible
                while ch % g != 0 and g > 1:
                    g -= 1
                return nn.GroupNorm(g, ch)
            return None

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            n = _norm(dims[i + 1])
            if n is not None:
                layers.append(n)
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierFeatures(nn.Module):
    """Fourier positional encoding for 2D inputs.

    Produces [sin, cos] features at exponentially increasing frequencies.
    If include_input=True, original (x,y) are concatenated as well.
    """

    def __init__(self, n_freqs: int, scale: float = 1.0, include_input: bool = True):
        super().__init__()
        self.n = max(0, int(n_freqs))
        self.scale = float(scale)
        self.include_input = bool(include_input)

    @property
    def out_dim(self) -> int:
        base = 2 if self.include_input else 0
        # For 2D coords: for each freq we add sin/cos for both x and y => 4 dims per freq
        return base + self.n * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., 2]
        if self.n == 0:
            return x if self.include_input else torch.zeros_like(x)
        orig_shape = x.shape
        x2 = x.view(-1, 2)
        feats: List[torch.Tensor] = []
        if self.include_input:
            feats.append(x2)
        for k in range(self.n):
            w = (2.0**k) * self.scale * math.pi
            s = torch.sin(w * x2)
            c = torch.cos(w * x2)
            feats.extend([s, c])
        y = torch.cat(feats, dim=-1)
        return y.view(*orig_shape[:-1], y.shape[-1])


class SetAbstraction(nn.Module):
    # Simple PointNet++ style set abstraction for 2D point sets
    def __init__(
        self,
        n_samples: int,
        radius: float,
        max_k: int,
        in_ch: int,
        out_ch: int,
        mlp_hidden: Optional[List[int]] = None,
        norm: str = "batch",
        num_groups: int = 16,
        pool: str = "max",  # 'max' | 'max+mean'
    ):
        super().__init__()
        self.n_samples = n_samples
        self.radius = radius
        self.max_k = max_k
        self.pool = pool
        if mlp_hidden is None:
            mlp_hidden = [out_ch // 2]
        self.mlp = MLP(in_ch + 2, mlp_hidden, out_ch, norm=norm, num_groups=num_groups)

    def forward(
        self, xyz: torch.Tensor, feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # xyz: [B,N,2]; feats: [B,N,C]
        B, N, _ = xyz.shape
        C = feats.shape[-1]
        # If n_samples <= 0 or >= N, treat all points as centers (no subsampling) for full coverage
        if self.n_samples <= 0 or self.n_samples >= N:
            centers = xyz  # [B,N,2]
            S = N
            idx = torch.arange(N, device=xyz.device).unsqueeze(0).repeat(B, 1)
        else:
            idx = farthest_point_sampling(xyz, self.n_samples)  # [B,S]
            centers = torch.gather(
                xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 2)
            )  # [B,S,2]
            S = centers.shape[1]
        knn_idx, knn_mask = ball_query(xyz, centers, self.radius, self.max_k)  # [B,S,K]
        grouped_xyz = torch.gather(
            xyz.unsqueeze(1).expand(B, S, N, 2),
            2,
            knn_idx.unsqueeze(-1).expand(-1, -1, -1, 2),
        )  # [B,S,K,2]
        grouped_feats = torch.gather(
            feats.unsqueeze(1).expand(B, S, N, C),
            2,
            knn_idx.unsqueeze(-1).expand(-1, -1, -1, C),
        )  # [B,S,K,C]
        rel = grouped_xyz - centers.unsqueeze(2)  # [B,S,K,2]
        x = torch.cat([rel, grouped_feats], dim=-1)  # [B,S,K,2+C]
        x = x.view(B * S * self.max_k, -1)
        x = self.mlp(x)
        x = x.view(B, S, self.max_k, -1)  # [B,S,K,OC]
        mask = knn_mask.unsqueeze(-1)
        # Use a large negative finite value for masked entries to avoid -inf propagation issues
        fill_val = -1e4
        x = torch.where(mask.expand_as(x), x, torch.full_like(x, fill_val))
        if self.pool == "max+mean":
            x_max = torch.max(x, dim=2).values  # [B,S,OC]
            # fill masked with 0 for mean
            x_masked = torch.where(mask.expand_as(x), x, torch.zeros_like(x))
            denom = torch.clamp(mask.sum(dim=2, keepdim=False).to(x.dtype), min=1.0)
            x_mean = x_masked.sum(dim=2) / denom  # [B,S,OC]
            x = torch.cat([x_max, x_mean], dim=-1)  # [B,S,2*OC]
        else:
            x = torch.max(x, dim=2).values  # [B,S,OC]
        return centers, x


class GlobalFeature(nn.Module):
    def __init__(
        self,
        in_ch: int,
        latent_dim: int,
        hidden: Optional[List[int]] = None,
        norm: str = "batch",
        num_groups: int = 16,
    ):
        super().__init__()
        if hidden is None:
            hidden = [max(8, latent_dim // 2)]
        self.mlp = MLP(in_ch, hidden, latent_dim, norm=norm, num_groups=num_groups)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        B, S, C = feats.shape
        x = feats.reshape(B * S, C)
        x = self.mlp(x)
        x = x.view(B, S, -1)
        x = torch.max(x, dim=1).values  # [B,latent]
        return x


class PointNet2Encoder2D(nn.Module):
    """Configurable PointNet++-style encoder for 2D point sets.

    encoder_cfg schema (defaults shown):
      {
        'latent_dim': 128,
        'pre_hidden': [64, 64],
        'sa_blocks': [
          {'n_samples': 256, 'radius': 0.10, 'max_k': 32, 'out_ch': 128, 'mlp_hidden': [64]},
          {'n_samples': 64,  'radius': 0.25, 'max_k': 32, 'out_ch': 256, 'mlp_hidden': [128]}
        ],
        'gf_hidden': [64]
      }
    """

    def __init__(
        self, latent_dim: int = 128, encoder_cfg: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        cfg = dict(encoder_cfg) if encoder_cfg is not None else {}
        self.latent_dim = int(cfg.get("latent_dim", latent_dim))
        pre_hidden: List[int] = list(cfg.get("pre_hidden", [64, 64]))
        sa_blocks_cfg: List[Dict[str, Any]] = list(
            cfg.get(
                "sa_blocks",
                [
                    {
                        "n_samples": 256,
                        "radius": 0.10,
                        "max_k": 32,
                        "out_ch": 128,
                        "mlp_hidden": [64],
                    },
                    {
                        "n_samples": 64,
                        "radius": 0.25,
                        "max_k": 32,
                        "out_ch": 256,
                        "mlp_hidden": [128],
                    },
                ],
            )
        )
        gf_hidden: List[int] = list(
            cfg.get("gf_hidden", [max(8, self.latent_dim // 2)])
        )
        # Normalization and pooling options
        norm_type: str = str(cfg.get("norm", "batch"))  # for SA and GF MLPs
        num_groups: int = int(cfg.get("num_groups", 16))
        pool_type: str = str(cfg.get("pool", "max"))  # 'max' | 'max+mean'

        # Optional Fourier positional encoding before pre-MLP
        posenc_cfg = cfg.get("posenc", None)
        self.posenc: Optional[FourierFeatures] = None
        in_ch_pre = 2
        if isinstance(posenc_cfg, dict):
            n_freqs = int(posenc_cfg.get("n_freqs", 0))
            scale = float(posenc_cfg.get("scale", 1.0))
            if n_freqs > 0:
                self.posenc = FourierFeatures(
                    n_freqs=n_freqs, scale=scale, include_input=True
                )
                in_ch_pre = self.posenc.out_dim

        # Pre pointwise MLP on coords
        pre_layers: List[nn.Module] = []
        dims = [in_ch_pre] + pre_hidden
        for i in range(len(dims) - 1):
            pre_layers.append(nn.Linear(dims[i], dims[i + 1]))
            pre_layers.append(nn.ReLU())
        self.pre = nn.Sequential(*pre_layers) if pre_layers else nn.Identity()
        current_in = dims[-1] if pre_layers else in_ch_pre

        # Set abstraction layers
        self.sa_layers = nn.ModuleList()
        for block in sa_blocks_cfg:
            n_samples = int(block.get("n_samples", 128))
            radius = float(block.get("radius", 0.1))
            max_k = int(block.get("max_k", 32))
            out_ch = int(block.get("out_ch", 128))
            mlp_hidden = block.get("mlp_hidden", None)
            if mlp_hidden is not None:
                mlp_hidden = [int(h) for h in mlp_hidden]
            sa = SetAbstraction(
                n_samples=n_samples,
                radius=radius,
                max_k=max_k,
                in_ch=current_in,
                out_ch=out_ch,
                mlp_hidden=mlp_hidden,
                norm=norm_type,
                num_groups=num_groups,
                pool=pool_type,
            )
            self.sa_layers.append(sa)
            current_in = out_ch * 2 if pool_type == "max+mean" else out_ch

        # Global feature aggregator
        self.glob = GlobalFeature(
            in_ch=current_in,
            latent_dim=self.latent_dim,
            hidden=gf_hidden,
            norm=norm_type,
            num_groups=num_groups,
        )

        # Persist resolved config
        self.encoder_cfg: Dict[str, Any] = {
            "latent_dim": self.latent_dim,
            "pre_hidden": pre_hidden,
            "sa_blocks": [
                {
                    "n_samples": int(b.get("n_samples", 128)),
                    "radius": float(b.get("radius", 0.1)),
                    "max_k": int(b.get("max_k", 32)),
                    "out_ch": int(b.get("out_ch", 128)),
                    "mlp_hidden": list(
                        b.get("mlp_hidden", [int(b.get("out_ch", 128)) // 2])
                    ),
                }
                for b in sa_blocks_cfg
            ],
            "gf_hidden": gf_hidden,
            "norm": norm_type,
            "num_groups": num_groups,
            "pool": pool_type,
        }
        if isinstance(posenc_cfg, dict):
            self.encoder_cfg["posenc"] = {
                "n_freqs": int(posenc_cfg.get("n_freqs", 0)),
                "scale": float(posenc_cfg.get("scale", 1.0)),
            }

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        x_in = self.posenc(xyz) if self.posenc is not None else xyz
        feats = self.pre(x_in)
        centers = xyz
        for sa in self.sa_layers:
            centers, feats = sa(centers, feats)
        latent = self.glob(feats)
        return latent


class PointNetMLPJoint(nn.Module):
    # Joint model: PointNet++ encoder + MLP head conditioned on query (x,y)
    def __init__(
        self,
        latent_dim: int = 128,
        mlp_hidden: Optional[List[int]] = None,
        encoder_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        # Encoder (cfg latent_dim overrides arg if provided)
        self.encoder = PointNet2Encoder2D(
            latent_dim=latent_dim, encoder_cfg=encoder_cfg
        )
        eff_latent = self.encoder.latent_dim
        if mlp_hidden is None:
            mlp_hidden = [256, 256, 128]
        self.head_hidden = list(mlp_hidden)
        # Optional Fourier features for query points in head
        head_posenc_cfg = None
        if encoder_cfg is not None:
            head_posenc_cfg = encoder_cfg.get("head_posenc", None)
        self.head_posenc: Optional[FourierFeatures] = None
        q_in_dim = 2
        if isinstance(head_posenc_cfg, dict):
            n_freqs = int(head_posenc_cfg.get("n_freqs", 0))
            scale = float(head_posenc_cfg.get("scale", 1.0))
            if n_freqs > 0:
                self.head_posenc = FourierFeatures(
                    n_freqs=n_freqs, scale=scale, include_input=True
                )
                q_in_dim = self.head_posenc.out_dim

        # Head normalization/dropout options
        head_norm = "batch"
        head_dropout = 0.0
        if encoder_cfg is not None:
            head_norm = str(encoder_cfg.get("head_norm", "batch"))
            head_dropout = float(encoder_cfg.get("head_dropout", 0.0))

        in_dim = eff_latent + q_in_dim
        self.head = MLP(
            in_dim,
            self.head_hidden,
            out_dim=1,
            norm=head_norm,
            num_groups=16,
            dropout=head_dropout,
        )

        # Persist arch for checkpoints
        enc_cfg_persist = dict(self.encoder.encoder_cfg)
        if isinstance(head_posenc_cfg, dict) and head_posenc_cfg.get("n_freqs", 0) > 0:
            enc_cfg_persist["head_posenc"] = {
                "n_freqs": int(head_posenc_cfg.get("n_freqs", 0)),
                "scale": float(head_posenc_cfg.get("scale", 1.0)),
            }
        # persist head normalization/dropout
        enc_cfg_persist["head_norm"] = head_norm
        enc_cfg_persist["head_dropout"] = head_dropout
        self._arch: Dict[str, Any] = {
            "encoder_cfg": enc_cfg_persist,
            "head_hidden": list(self.head_hidden),
        }

    def get_arch(self) -> Dict[str, Any]:
        # Return the persisted architecture (including optional head_posenc if present).
        # self._arch was constructed in __init__ with encoder_cfg (plus head_posenc when used)
        return dict(self._arch)

    def forward(
        self, geom_points: torch.Tensor, query_points: torch.Tensor
    ) -> torch.Tensor:
        z = self.encoder(geom_points)  # [B,L]
        B, Q, _ = query_points.shape
        q_feat = (
            self.head_posenc(query_points)
            if self.head_posenc is not None
            else query_points
        )
        z_exp = z.unsqueeze(1).expand(-1, Q, -1)
        x = torch.cat([z_exp, q_feat], dim=-1)
        x = x.reshape(B * Q, -1)
        y = self.head(x)
        y = y.view(B, Q, 1)
        return y


def build_model_from_arch(arch: Dict[str, Any]) -> PointNetMLPJoint:
    """Reconstruct a PointNetMLPJoint from an 'arch' dict saved in a checkpoint.

    Expected keys:
        arch = {
            'encoder_cfg': {...},
            'head_hidden': [...]
        }
    """
    encoder_cfg = arch.get("encoder_cfg", None)
    head_hidden = arch.get("head_hidden", [256, 256, 128])
    latent_dim = int(encoder_cfg.get("latent_dim", 128)) if encoder_cfg else 128
    return PointNetMLPJoint(
        latent_dim=latent_dim, mlp_hidden=head_hidden, encoder_cfg=encoder_cfg
    )
