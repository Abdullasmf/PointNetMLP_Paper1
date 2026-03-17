import torch
import torch.nn as nn
import math
from typing import List, Dict, Any, Optional, Tuple, Union

# Import necessary components from pn_models
from pn_models import PointNet2Encoder2D, MLP, farthest_point_sampling, ball_query

class FourierFeatures(nn.Module):
    """Fourier positional encoding for 2D inputs.
    
    Produces [sin, cos] features at exponentially increasing frequencies.
    If include_input=True, original (x,y) are concatenated as well.
    (Re-implemented here to ensure self-contained usage for the benchmarks)
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


class VanillaDeepONet(nn.Module):
    """
    Model 1: VanillaDeepONet
    Architecture: Standard DeepONet (Branch + Trunk + Dot Product)
    Branche: PointNet2Encoder2D -> MLP -> coefficients b
    Trunk: Raw (x,y) -> MLP -> basis functions t
    Fusion: Dot product + bias
    Constraint: No Fourier Features
    """
    def __init__(
        self,
        latent_dim: int = 128,
        basis_dim: int = 128,
        branch_hidden: List[int] = [256, 256],
        trunk_hidden: List[int] = [256, 256],
        out_dim: int = 1,
        encoder_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        # Extract normalization settings from encoder_cfg
        norm_type = "batch"  # default
        num_groups = 16  # default
        if encoder_cfg is not None:
            norm_type = str(encoder_cfg.get("norm", "batch"))
            num_groups = int(encoder_cfg.get("num_groups", 16))
        
        # 1. Branch Network
        # Shared Encoder
        self.encoder = PointNet2Encoder2D(latent_dim=latent_dim, encoder_cfg=encoder_cfg)
        eff_latent = self.encoder.latent_dim
        
        # Branch MLP: maps latent vector z to basis coefficients b
        # Output dim is basis_dim * out_dim. We will reshape later if out_dim > 1.
        self.branch_net = MLP(
            in_dim=eff_latent,
            hidden=branch_hidden,
            out_dim=basis_dim * out_dim,
            norm=norm_type,
            num_groups=num_groups
        )

        # 2. Trunk Network
        # Input: Raw (x,y) -> 2 dims
        # Output: basis functions t
        self.trunk_net = MLP(
            in_dim=2,
            hidden=trunk_hidden,
            out_dim=basis_dim * out_dim,
            norm=norm_type,
            num_groups=num_groups
        )
        
        self.basis_dim = basis_dim
        self.out_dim = out_dim
        
        # 3. Bias
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, geom_points: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
        """
        geom_points: [B, N, 2] - Geometry input for Branch
        query_points: [B, Q, 2] - Query points for Trunk
        Output: [B, Q, Out_Dim]
        """
        B, Q, _ = query_points.shape

        # Branch execution
        z = self.encoder(geom_points) # [B, latent_dim]
        coeffs = self.branch_net(z)   # [B, basis_dim * out_dim]
        
        # Reshape coeffs to [B, 1, basis_dim, out_dim] to broadcast over Q
        coeffs = coeffs.view(B, 1, self.basis_dim, self.out_dim)

        # Trunk execution
        # Flatten query_points to [B*Q, 2] for MLP processing (BatchNorm1d compatibility)
        query_flat = query_points.reshape(B * Q, -1)  # [B*Q, 2]
        basis_flat = self.trunk_net(query_flat)  # [B*Q, basis_dim * out_dim]
        
        # Reshape basis to [B, Q, basis_dim, out_dim]
        basis = basis_flat.view(B, Q, self.basis_dim, self.out_dim)

        # Fusion: Dot product over basis_dim
        # Sum( b_i * t_i )
        # coeffs: [B, 1, P, O]
        # basis:  [B, Q, P, O]
        # product: [B, Q, P, O] -> sum(dim=2) -> [B, Q, O]
        out = torch.sum(coeffs * basis, dim=2)
        
        # Add bias
        out = out + self.bias
        
        return out


class SpectralDeepONet(nn.Module):
    """
    Model 2: SpectralDeepONet
    Architecture: Similar to VanillaDeepONet
    Difference: Trunk input passes through FourierFeature mapping first.
    """
    def __init__(
        self,
        latent_dim: int = 128,
        basis_dim: int = 128,
        branch_hidden: List[int] = [256, 256],
        trunk_hidden: List[int] = [256, 256],
        out_dim: int = 1,
        n_freqs: int = 10,
        scale: float = 2.0,
        encoder_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        # Extract normalization settings from encoder_cfg
        norm_type = "batch"  # default
        num_groups = 16  # default
        if encoder_cfg is not None:
            norm_type = str(encoder_cfg.get("norm", "batch"))
            num_groups = int(encoder_cfg.get("num_groups", 16))
        
        # 1. Branch Network
        self.encoder = PointNet2Encoder2D(latent_dim=latent_dim, encoder_cfg=encoder_cfg)
        eff_latent = self.encoder.latent_dim
        
        self.branch_net = MLP(
            in_dim=eff_latent,
            hidden=branch_hidden,
            out_dim=basis_dim * out_dim,
            norm=norm_type,
            num_groups=num_groups
        )

        # 2. Trunk Network
        # FFT Mapping first
        self.fourier = FourierFeatures(n_freqs=n_freqs, scale=scale, include_input=True)
        trunk_in_dim = self.fourier.out_dim
        
        self.trunk_net = MLP(
            in_dim=trunk_in_dim,
            hidden=trunk_hidden,
            out_dim=basis_dim * out_dim,
            norm=norm_type,
            num_groups=num_groups
        )
        
        self.basis_dim = basis_dim
        self.out_dim = out_dim
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, geom_points: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
        B, Q, _ = query_points.shape

        # Branch
        z = self.encoder(geom_points) # [B, latent_dim]
        coeffs = self.branch_net(z)   # [B, basis_dim * out_dim]
        coeffs = coeffs.view(B, 1, self.basis_dim, self.out_dim)

        # Trunk
        # Apply FFT to query points
        q_embed = self.fourier(query_points) # [B, Q, trunk_in_dim]
        # Flatten for MLP processing (BatchNorm1d compatibility)
        q_flat = q_embed.reshape(B * Q, -1)  # [B*Q, trunk_in_dim]
        basis_flat = self.trunk_net(q_flat)  # [B*Q, basis_dim * out_dim]
        basis = basis_flat.view(B, Q, self.basis_dim, self.out_dim)

        # Fusion
        out = torch.sum(coeffs * basis, dim=2)
        out = out + self.bias
        
        return out


class DenseNoFFT(nn.Module):
    """
    Model 3: DenseNoFFT
    Architecture: Identical to PointNetMLPJoint (Concatenation decoder)
    Constraint: Strictly NO FFT (raw x,y coordinates only).
    """
    def __init__(
        self,
        latent_dim: int = 128,
        mlp_hidden: List[int] = [256, 256, 128],
        out_dim: int = 1,
        encoder_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        # Extract normalization settings from encoder_cfg
        norm_type = "batch"  # default
        num_groups = 16  # default
        if encoder_cfg is not None:
            norm_type = str(encoder_cfg.get("norm", "batch"))
            num_groups = int(encoder_cfg.get("num_groups", 16))
        
        # Shared Encoder
        self.encoder = PointNet2Encoder2D(latent_dim=latent_dim, encoder_cfg=encoder_cfg)
        eff_latent = self.encoder.latent_dim
        
        # Decoder (Head)
        # Input: latent z (dim=eff_latent) + query (x,y) (dim=2)
        head_in_dim = eff_latent + 2
        
        # We append out_dim to the hidden list or usage logic from PointNetMLPJoint
        # PointNetMLPJoint uses a dedicated MLP and hardcodes output to 1.
        # We must support generic out_dim as per requirements.
        
        self.head = MLP(
            in_dim=head_in_dim,
            hidden=mlp_hidden,
            out_dim=out_dim,
            norm=norm_type,
            num_groups=num_groups
            # PointNetMLPJoint uses dropout and norm from config, effectively logic duplicated here for simplicity
        )

    def forward(self, geom_points: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
        """
        Concatenation based decoder without FFT.
        """
        # Encoder
        z = self.encoder(geom_points) # [B, L]
        
        B, Q, _ = query_points.shape
        
        # Expand latent to match queries
        z_exp = z.unsqueeze(1).expand(-1, Q, -1) # [B, Q, L]
        
        # Concatenate: [z, x, y]
        # query_points is [B, Q, 2]
        x_in = torch.cat([z_exp, query_points], dim=-1) # [B, Q, L+2]
        
        # Pass through Decoder MLP
        # The MLP class uses Sequential, which supports multidimensional input if Linear layers are used.
        # However, PointNetMLPJoint flattens B*Q before MLP. I'll do the same to be safe and identical to reference.
        x_flat = x_in.reshape(B * Q, -1)
        y_flat = self.head(x_flat)
        
        # Reshape back
        y = y_flat.view(B, Q, -1) # [B, Q, Out_Dim]
        
        return y


class FourierFeatureMapping(nn.Module):
    """Fourier Feature Mapping (FFM) for arbitrary-dimensional inputs.

    Given input v of shape [..., in_dim]:
      1. Project: proj = v @ B_mat                   [..., ffm_mapping_size]
      2. Scale:   scaled = 2*pi * proj * sigma        [..., ffm_mapping_size]
      3. Encode:  gamma(v) = [cos(scaled), sin(scaled)] [..., 2*ffm_mapping_size]

    The random Gaussian matrix B_mat is registered as a *buffer* (frozen, requires_grad=False).
    The scale sigma is a learnable nn.Parameter.
    """

    def __init__(self, in_dim: int, ffm_mapping_size: int, sigma_init: float = 10.0):
        super().__init__()
        # Frozen random projection matrix: [in_dim, ffm_mapping_size]
        B_mat = torch.randn(in_dim, ffm_mapping_size)
        self.register_buffer("B_mat", B_mat)  # NOT a parameter – strictly frozen
        # Learnable scale
        self.sigma = nn.Parameter(torch.tensor(float(sigma_init)))

    @property
    def out_dim(self) -> int:
        return 2 * self.B_mat.shape[1]  # cos + sin concatenated

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v: [..., in_dim]
        proj = v @ self.B_mat                          # [..., ffm_mapping_size]
        scaled = 2.0 * math.pi * proj * self.sigma     # [..., ffm_mapping_size]
        return torch.cat([torch.cos(scaled), torch.sin(scaled)], dim=-1)  # [..., 2*ffm_mapping_size]


class ScaledDiagramDeepONet(nn.Module):
    """Refactored ScaledDiagramDeepONet with FFM trunk and Cross-Attention bridge.

    Branch Mutation:
        PointNet2Encoder2D is configured with return_local_features=True so that it
        returns a tuple (local_feats [B, N', latent_dim], centers [B, N', 2]) where
        centers are the 2D physical coordinates of the subsampled SA points.

    Trunk Mutation:
        SIREN is replaced by a Fourier Feature Mapping (FFM) module followed by a
        standard MLP with SiLU activations that projects to basis_dim.
        Input v = (x, y, SDF): [B, Q, 3].

    Bridge Mutation:
        Element-wise multiplication is replaced by spatially-grounded, temperature-
        scaled Multi-Head Cross-Attention.
        - centers [B, N', 2] are encoded via center_ffm + center_mlp -> center_pos_embed [B, N', basis_dim]
        - K = kv_proj(local_feats) + center_pos_embed  (spatially grounded keys)
        - V = kv_proj(local_feats)                     (unmodified values)
        - Q_scaled = t_beta / attn_temp                (temperature-scaled queries)
        The attended output is passed through a post-attention MLP to produce B^beta.

    Terminal:
        DeepONet dot product: sum(B^beta * T^beta, dim=-1) + bias -> [B, Q, 1].
    """

    def __init__(
        self,
        latent_dim: int = 192,
        basis_dim: int = 128,
        head_hidden: List[int] = [512, 512, 256],
        ffm_mapping_size: int = 128,
        ffm_sigma_init: float = 2.0,
        cross_attention_heads: int = 4,
        attn_temp: float = 0.1,
        encoder_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        # SDF channel count (0 = no SDF, 1 = SDF appended as 3rd per-point feature)
        sdf_ch = int(encoder_cfg.get("sdf_ch", 0)) if encoder_cfg else 0
        self.sdf_ch = sdf_ch
        trunk_in_ch = 2 + sdf_ch  # FFM input dimensionality: (x, y) or (x, y, sdf)

        norm_type = str(encoder_cfg.get("norm", "batch")) if encoder_cfg else "batch"
        num_groups = int(encoder_cfg.get("num_groups", 16)) if encoder_cfg else 16

        # Attention temperature: scales Q before cross-attention to control softmax sharpness
        self.attn_temp = float(attn_temp)

        # ------------------------------------------------------------------ #
        # 1. Branch Network: PointNet++ returning local features [B, N', latent_dim]
        #    and spatial centers [B, N', 2]
        # ------------------------------------------------------------------ #
        # Inject return_local_features=True so the encoder returns (local_feats, centers).
        enc_cfg: Dict[str, Any] = dict(encoder_cfg) if encoder_cfg is not None else {}
        enc_cfg["return_local_features"] = True
        self.encoder = PointNet2Encoder2D(latent_dim=latent_dim, encoder_cfg=enc_cfg)
        eff_latent = self.encoder.latent_dim  # == latent_dim after glob.mlp projection

        # ------------------------------------------------------------------ #
        # 2. Trunk Network: FFM + SiLU MLP -> [B, Q, basis_dim]
        # ------------------------------------------------------------------ #
        self.ffm = FourierFeatureMapping(
            in_dim=trunk_in_ch,
            ffm_mapping_size=ffm_mapping_size,
            sigma_init=ffm_sigma_init,
        )
        ffm_out_dim = self.ffm.out_dim  # 2 * ffm_mapping_size

        # MLP with SiLU activations: ffm_out_dim -> basis_dim
        self.trunk_mlp = MLP(
            in_dim=ffm_out_dim,
            hidden=head_hidden,
            out_dim=basis_dim,
            act=nn.SiLU,
            norm=norm_type,
            num_groups=num_groups,
        )

        # ------------------------------------------------------------------ #
        # 3. Bridge: Spatially Grounded, Temperature-Scaled Cross-Attention
        # ------------------------------------------------------------------ #
        # Linear projection to align branch latent_dim -> basis_dim for K and V
        self.kv_proj = nn.Linear(eff_latent, basis_dim)

        # Center positional encoding: FFM for the 2D SA centers
        self.center_ffm = FourierFeatureMapping(
            in_dim=2,
            ffm_mapping_size=ffm_mapping_size,
            sigma_init=ffm_sigma_init,
        )
        # MLP to project center FFM output [2*ffm_mapping_size] -> basis_dim
        self.center_mlp = MLP(
            in_dim=2 * ffm_mapping_size,
            hidden=head_hidden,
            out_dim=basis_dim,
            act=nn.SiLU,
            norm=norm_type,
            num_groups=num_groups,
        )

        # Validate that basis_dim is divisible by cross_attention_heads
        assert basis_dim % cross_attention_heads == 0, (
            f"basis_dim ({basis_dim}) must be divisible by cross_attention_heads ({cross_attention_heads})"
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=basis_dim,
            num_heads=cross_attention_heads,
            batch_first=True,
        )

        # Post-attention MLP: produces B^beta [B, Q, basis_dim]
        self.post_attn_mlp = MLP(
            in_dim=basis_dim,
            hidden=head_hidden,
            out_dim=basis_dim,
            norm=norm_type,
            num_groups=num_groups,
        )

        self.basis_dim = basis_dim
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, geom_points: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            geom_points:  [B, N, 2+sdf_ch]  – geometry point cloud with optional SDF
            query_points: [B, Q, 2+sdf_ch]  – query locations (x, y[, sdf])

        Returns:
            [B, Q, 1]  – predicted scalar stress at each query point
        """
        B, Q, q_ch = query_points.shape

        # ------------------------------------------------------------------ #
        # BRANCH: encode geometry into discrete local features + SA centers
        # geom_points [B, N, 2+sdf_ch]
        #   -> local_feats [B, N', eff_latent]
        #   -> centers     [B, N', 2]
        # ------------------------------------------------------------------ #
        local_feats, centers = self.encoder(geom_points)  # [B, N', eff_latent], [B, N', 2]
        N_prime = local_feats.shape[1]

        # ------------------------------------------------------------------ #
        # TRUNK: Fourier Feature Mapping + SiLU MLP
        # query_points [B, Q, 2+sdf_ch]
        #   -> ffm_out  [B, Q, 2*ffm_mapping_size]
        #   -> t_beta   [B, Q, basis_dim]
        # ------------------------------------------------------------------ #
        ffm_out = self.ffm(query_points)                  # [B, Q, 2*ffm_mapping_size]
        ffm_flat = ffm_out.reshape(B * Q, -1)             # [B*Q, 2*ffm_mapping_size]
        t_beta = self.trunk_mlp(ffm_flat).view(B, Q, self.basis_dim)  # [B, Q, basis_dim]

        # ------------------------------------------------------------------ #
        # BRIDGE: Spatially Grounded, Temperature-Scaled Cross-Attention
        #
        # Positional encoding for SA centers:
        #   centers [B, N', 2]
        #   -> center_ffm_out    [B, N', 2*ffm_mapping_size]
        #   -> center_pos_embed  [B, N', basis_dim]
        #
        # Spatially grounded Keys:
        #   V = kv_proj(local_feats)                     [B, N', basis_dim]
        #   K = kv_proj(local_feats) + center_pos_embed  [B, N', basis_dim]
        #
        # Temperature-scaled Queries:
        #   Q_scaled = t_beta / attn_temp                [B, Q, basis_dim]
        # ------------------------------------------------------------------ #
        center_ffm_out = self.center_ffm(centers)                                        # [B, N', 2*ffm_mapping_size]
        center_ffm_flat = center_ffm_out.reshape(B * N_prime, self.center_ffm.out_dim)  # [B*N', 2*ffm_mapping_size]
        center_pos_embed = self.center_mlp(center_ffm_flat).view(
            B, N_prime, self.basis_dim
        )                                                                 # [B, N', basis_dim]

        V = self.kv_proj(local_feats)                                    # [B, N', basis_dim]
        K = V + center_pos_embed                                         # [B, N', basis_dim]

        Q_scaled = t_beta / self.attn_temp                               # [B, Q, basis_dim]

        # cross_attn(query, key, value) with batch_first=True
        attended, _ = self.cross_attn(Q_scaled, K, V)                   # [B, Q, basis_dim]

        # Post-attention MLP: produces B^beta
        attended_flat = attended.reshape(B * Q, -1)                      # [B*Q, basis_dim]
        b_beta = self.post_attn_mlp(attended_flat).view(
            B, Q, self.basis_dim
        )                                                                 # [B, Q, basis_dim]

        # ------------------------------------------------------------------ #
        # TERMINAL DOT PRODUCT: sum(B^beta * T^beta, dim=-1) + bias
        # b_beta: [B, Q, basis_dim]
        # t_beta: [B, Q, basis_dim]
        # out:    [B, Q, 1]
        # ------------------------------------------------------------------ #
        out = torch.sum(b_beta * t_beta, dim=-1, keepdim=True) + self.bias  # [B, Q, 1]

        return out

# ===========================================================================
# ArGEnT DeepONet  –  Galerkin linear attention operator network
# Supports two variants:
#   attention_type='cross', use_sdf=True   (ArGEnT cross-attention with SDF)
#   attention_type='self',  use_sdf=False  (ArGEnT self-attention, no SDF)
# ===========================================================================

class _PointwiseMLP(nn.Module):
    """Point-wise MLP: n_hidden hidden layers, hidden_dim neurons each, ReLU.

    No batch normalization, no dropout.
    Input: [..., in_dim]  →  Output: [..., hidden_dim]
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_hidden: int = 4) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        out = self.net(x.reshape(-1, orig_shape[-1]))
        return out.reshape(*orig_shape[:-1], out.shape[-1])


class _GalerkinAttentionLayer(nn.Module):
    """Single Galerkin-type linear attention layer with 2-D RoPE.

    Formulation (per head):
        K̃ = LayerNorm(K),  Ṽ = LayerNorm(V)
        out = Q (K̃ᵀ Ṽ) / n
    where n is the number of key / value elements.

    Rotary Position Embeddings (RoPE) are applied to Q and K before attention.
    No dropout anywhere.
    """

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        head_dim = hidden_dim // num_heads
        assert head_dim % 4 == 0, (
            f"head_dim ({head_dim}) must be divisible by 4 for 2-D RoPE"
        )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Layer norms for Galerkin formulation (K̃, Ṽ)
        self.k_norm = nn.LayerNorm(head_dim)
        self.v_norm = nn.LayerNorm(head_dim)

    def _apply_rope_2d(
        self, x: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """Apply 2-D Rotary Position Embeddings.

        Args:
            x      : [B, S, H, D]  (B=batch, S=seq len, H=heads, D=head_dim)
            coords : [B, S, 2]     spatial (x, y) coordinates

        Returns: rotated tensor of same shape.
        """
        B, S, H, D = x.shape
        quarter_D = D // 4
        device, dtype = x.device, x.dtype

        i = torch.arange(quarter_D, device=device, dtype=dtype)
        freq = 1.0 / (10000.0 ** (2.0 * i / quarter_D))  # [D/4]

        # [B, S, D/4]
        theta_x = coords[..., 0].unsqueeze(-1) * freq
        theta_y = coords[..., 1].unsqueeze(-1) * freq

        # [B, S, 1, D/4]  – broadcast over heads
        cos_x, sin_x = theta_x.cos().unsqueeze(2), theta_x.sin().unsqueeze(2)
        cos_y, sin_y = theta_y.cos().unsqueeze(2), theta_y.sin().unsqueeze(2)

        # Split into four equal quarters
        x1, x2 = x[..., :quarter_D], x[..., quarter_D: 2 * quarter_D]
        x3, x4 = x[..., 2 * quarter_D: 3 * quarter_D], x[..., 3 * quarter_D:]

        # First D/2: rotate with x-coordinate
        x1_r = x1 * cos_x - x2 * sin_x
        x2_r = x1 * sin_x + x2 * cos_x
        # Second D/2: rotate with y-coordinate
        x3_r = x3 * cos_y - x4 * sin_y
        x4_r = x3 * sin_y + x4 * cos_y

        return torch.cat([x1_r, x2_r, x3_r, x4_r], dim=-1)  # [B, S, H, D]

    def forward(
        self,
        q_feat: torch.Tensor,
        kv_feat: torch.Tensor,
        q_coords: torch.Tensor,
        kv_coords: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
        q_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q_feat   : [B, S_q,  hidden_dim]
            kv_feat  : [B, S_kv, hidden_dim]
            q_coords : [B, S_q,  2]
            kv_coords: [B, S_kv, 2]
            kv_mask  : [B, S_kv] bool, True=real point, False=zero-padded (optional)
            q_mask   : [B, S_q]  bool, True=real point, False=zero-padded (optional)
        Returns: [B, S_q, hidden_dim]
        """
        B, S_q, _ = q_feat.shape
        S_kv = kv_feat.shape[1]
        H, D = self.num_heads, self.head_dim

        # Project → [B, S, H, D]
        Q = self.q_proj(q_feat).view(B, S_q, H, D)
        K = self.k_proj(kv_feat).view(B, S_kv, H, D)
        V = self.v_proj(kv_feat).view(B, S_kv, H, D)

        # Apply 2-D RoPE to Q and K
        Q = self._apply_rope_2d(Q, q_coords)
        K = self._apply_rope_2d(K, kv_coords)

        # Zero out padded K and V positions before LayerNorm
        if kv_mask is not None:
            kv_mask_feat = kv_mask.unsqueeze(-1).unsqueeze(-1).to(K.dtype)  # [B, S_kv, 1, 1]
            K = K * kv_mask_feat
            V = V * kv_mask_feat

        # Galerkin: normalise K and V
        K_n = self.k_norm(K)  # [B, S_kv, H, D]
        V_n = self.v_norm(V)  # [B, S_kv, H, D]

        # Zero out padded K_n and V_n positions after LayerNorm
        if kv_mask is not None:
            K_n = K_n * kv_mask_feat
            V_n = V_n * kv_mask_feat

        # Zero out padded Q positions
        if q_mask is not None:
            q_mask_feat = q_mask.unsqueeze(-1).unsqueeze(-1).to(Q.dtype)  # [B, S_q, 1, 1]
            Q = Q * q_mask_feat

        # Reshape to [B*H, S, D] for batched matmul
        Q_bh = Q.permute(0, 2, 1, 3).reshape(B * H, S_q, D)
        K_bh = K_n.permute(0, 2, 1, 3).reshape(B * H, S_kv, D)
        V_bh = V_n.permute(0, 2, 1, 3).reshape(B * H, S_kv, D)

        # True number of valid key/value positions per batch item
        if kv_mask is not None:
            n = kv_mask.sum(dim=1).float().clamp(min=1.0)  # [B]
            # Expand to [B*H, 1, 1] for broadcasting over the D×D KV matrix
            n_bh = n.unsqueeze(1).expand(B, H).reshape(B * H).view(B * H, 1, 1)
        else:
            n_bh = float(S_kv)

        # KV = Kᵀ V / n  →  [B*H, D, D]
        KV = torch.bmm(K_bh.transpose(-2, -1), V_bh) / n_bh

        # out = Q KV  →  [B*H, S_q, D]
        out = torch.bmm(Q_bh, KV)

        # Merge heads → [B, S_q, hidden_dim]
        out = out.view(B, H, S_q, D).permute(0, 2, 1, 3).reshape(B, S_q, H * D)
        out = self.out_proj(out)

        # Zero out padded output positions so they remain zero for the next layer
        if q_mask is not None:
            out = out * q_mask.unsqueeze(-1).to(out.dtype)  # [B, S_q, hidden_dim]

        return out


class ArGEnTDeepONet(nn.Module):
    """ArGEnT – Galerkin linear attention operator network (no branch network).

    The model encodes query coordinates directly through the Trunk (ArGEnT attention
    layers + residual output MLP) and projects the result to a scalar stress prediction.
    There is no branch network and no DeepONet dot-product fusion.

    Variants
    --------
    attention_type='cross', use_sdf=True
        Geometric point cloud (x, y, sdf) provides Keys / Values.
        Query coordinates (x, y, sdf) are projected as Queries.
    attention_type='self', use_sdf=False
        Query coordinates (x, y) only; Q = K = V from the same projected input.
        geom_points is not processed.

    All projector MLPs: 4 hidden layers, hidden_dim neurons, ReLU.
    Residual output MLP: 3 hidden layers, hidden_dim neurons, ReLU.
    No dropout or batch normalization anywhere.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        output_dim: int = 128,
        attention_type: str = "cross",
        use_sdf: bool = True,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        assert (hidden_dim // num_heads) % 4 == 0, (
            f"head_dim ({hidden_dim // num_heads}) must be divisible by 4 for 2-D RoPE"
        )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.attention_type = attention_type
        self.use_sdf = use_sdf

        # Input channel counts
        in_ch_query = 2  # self-attention: (x, y) only, no SDF

        # ── TRUNK (ArGEnT) ──────────────────────────────────────────────────
        self.proj_mlp = _PointwiseMLP(in_ch_query, hidden_dim, n_hidden=4)

        self.attn_layers = nn.ModuleList(
            [_GalerkinAttentionLayer(hidden_dim, num_heads) for _ in range(num_layers)]
        )

        # Residual output MLP – 3 hidden layers
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

        # Final projection: hidden_dim → scalar stress prediction
        self.trunk_proj = nn.Linear(hidden_dim, 1)

        self.bias = nn.Parameter(torch.zeros(1))

    # -----------------------------------------------------------------------
    def get_arch(self) -> dict:
        return {
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "output_dim": self.output_dim,
            "attention_type": self.attention_type,
            "use_sdf": self.use_sdf,
        }

    def forward(
        self,
        geom_points: torch.Tensor,   # [B, N, in_ch_geom] – not used for self-attention
        query_points: torch.Tensor,  # [B, Q, 2]
        mask: Optional[torch.Tensor] = None,  # [B, Q] bool: True=real, False=zero-padded
    ) -> torch.Tensor:
        """Returns [B, Q, 1]."""
        B, Q_len, _ = query_points.shape

        # Spatial (x, y) coordinates for RoPE
        query_coords = query_points[..., :2]  # [B, Q, 2]

        # ── TRUNK (self-attention; geom_points is not used) ─────────────────
        q = self.proj_mlp(query_points)  # [B, Q, hidden_dim]

        # Zero out padded positions at the start
        if mask is not None:
            q = q * mask.unsqueeze(-1).to(q.dtype)

        for attn in self.attn_layers:
            q = q + attn(q, q, query_coords, query_coords, kv_mask=mask, q_mask=mask)

        # Residual output MLP
        q_res = self.out_mlp(q.reshape(-1, self.hidden_dim)).view(B, Q_len, self.hidden_dim)
        if mask is not None:
            q_res = q_res * mask.unsqueeze(-1).to(q_res.dtype)
        q = q + q_res

        # Final projection to scalar stress value
        out = self.trunk_proj(q.reshape(-1, self.hidden_dim)).view(B, Q_len, 1) + self.bias
        # Zero out padded positions in the final output
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


# ===========================================================================
# GINOT – Geometry-Informed Neural Operator with Transformers
#
# Architecture:
#   1. GINOTGeometryEncoder: encodes boundary point cloud into latent K and V
#      - FPS + ball query grouping
#      - NeRF positional encoding + local feature aggregation
#      - Cross-attention (local features as Q, global features as K/V)
#      - Self-attention refinement
#
#   2. GINOTDecoder: decodes query points into solution field
#      - NeRF positional encoding + MLP -> Q matrix
#      - Cross-attention using encoder K and V
#      - Output MLP -> scalar prediction
#
#   3. GINOT: top-level class combining encoder and decoder
# ===========================================================================


class _GINOTAttentionBlock(nn.Module):
    """Post-norm Transformer block supporting both self- and cross-attention.

    For cross-attention, k and v are provided externally (encoder output).
    For self-attention, k and v are None (uses q as both key and value).

    Uses standard softmax attention (not Galerkin linear attention) so that
    key_padding_mask can force padded positions to -inf.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(
        self,
        q: torch.Tensor,                            # [B, Sq, d_model]
        k: Optional[torch.Tensor] = None,           # [B, Skv, d_model] or None (self-attn)
        v: Optional[torch.Tensor] = None,           # [B, Skv, d_model] or None (self-attn)
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, Skv] True=ignore
    ) -> torch.Tensor:
        """Returns [B, Sq, d_model]."""
        if k is None:
            k = q  # self-attention
        if v is None:
            v = q  # self-attention
        attn_out, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask)
        x = self.norm1(q + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class GINOTGeometryEncoder(nn.Module):
    """GINOT Geometry Encoder.

    Encodes a (possibly padded) boundary point cloud into latent KEY and VALUE
    matrices of shape B x Ns x d_model for use by the decoder.

    Steps
    -----
    1. Apply NeRF positional encoding + linear projection to every point
       -> global feature map  [B, N, d_model]
    2. Farthest Point Sampling -> Ns centroid indices  [B, Ns]
    3. Ball-query grouping around each centroid -> [B, Ns, Np, ...]
    4. Index global feature map with group indices, concatenate relative
       coordinates, pass through MLP, max-pool over Np
       -> local aggregated features  [B, Ns, d_model]
    5. num_encoder_cross_layers Cross-Attention blocks:
       Q = local features, K = V = global features
       (padded global points masked out via key_padding_mask)
    6. num_encoder_self_layers Self-Attention blocks on the result
    7. Linear projections -> encoder K and V  [B, Ns, d_model]
    """

    def __init__(
        self,
        d_in: int = 2,
        d_model: int = 128,
        n_s: Union[int, float] = 128,
        n_p: int = 32,
        radius: float = 0.1,
        n_heads: int = 4,
        num_cross_layers: int = 2,
        num_self_layers: int = 2,
        n_freqs: int = 6,
        mlp_hidden_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [d_model * 2, d_model * 2]

        self.n_s = n_s
        self.n_p = n_p
        self.radius = radius
        self.d_model = d_model

        # NeRF positional encoding (FourierFeatures already defined in this module)
        self.posenc = FourierFeatures(n_freqs=n_freqs, scale=1.0, include_input=True)
        posenc_dim = self.posenc.out_dim  # 2 + n_freqs * 4

        # Linear projection: posenc_dim -> d_model  (global feature map)
        self.feat_proj = nn.Linear(posenc_dim, d_model)

        # MLP for group aggregation: (d_model + d_in) per neighbour -> d_model
        self.group_mlp = MLP(
            in_dim=d_model + d_in,
            hidden=mlp_hidden_dims,
            out_dim=d_model,
            act=nn.GELU,
            norm="layer",
        )

        # Cross-attention blocks: local features (Q) x global features (K, V)
        self.cross_attn_blocks = nn.ModuleList(
            [_GINOTAttentionBlock(d_model, n_heads) for _ in range(num_cross_layers)]
        )

        # Self-attention blocks on the Ns centroid tokens
        self.self_attn_blocks = nn.ModuleList(
            [_GINOTAttentionBlock(d_model, n_heads) for _ in range(num_self_layers)]
        )

        # Output linear projections -> encoder K and V
        self.k_out = nn.Linear(d_model, d_model)
        self.v_out = nn.Linear(d_model, d_model)

    def forward(
        self,
        geom_points: torch.Tensor,           # [B, N, d_in]
        mask: Optional[torch.Tensor] = None, # [B, N] bool, True=real point
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (enc_k, enc_v) each of shape [B, Ns, d_model]."""
        B, N, d = geom_points.shape

        # ------------------------------------------------------------------
        # 1. Global feature map via NeRF posenc + linear projection
        # ------------------------------------------------------------------
        posenc_out = self.posenc(geom_points)          # [B, N, posenc_dim]
        global_feat = self.feat_proj(posenc_out)        # [B, N, d_model]

        # ------------------------------------------------------------------
        # 2. Farthest Point Sampling -> Ns centroid indices
        #    Skip FPS when n_s <= 0 or n_s >= N (use all points as centroids)
        #    If n_s is a float in (0.0, 1.0], treat it as a sampling ratio.
        # ------------------------------------------------------------------
        if 0.0 < self.n_s <= 1.0:
            if mask is not None:
                max_valid = int(mask.sum(dim=1).max().item())
            else:
                max_valid = N
            effective_n_s = max(1, int(self.n_s * max_valid))
        else:
            effective_n_s = int(self.n_s)

        if effective_n_s <= 0 or effective_n_s >= N:
            effective_n_s = N
            centroids = geom_points  # [B, N, d_in]
        else:
            fps_idx = farthest_point_sampling(geom_points, effective_n_s)  # [B, n_s]
            # Gather centroid coordinates
            centroids = torch.gather(
                geom_points,
                1,
                fps_idx.unsqueeze(-1).expand(-1, -1, d),
            )  # [B, n_s, d_in]

        # ------------------------------------------------------------------
        # 3. Ball-query grouping
        # ------------------------------------------------------------------
        # knn_idx: [B, n_s, n_p]  knn_mask: [B, n_s, n_p] True=valid
        knn_idx, knn_mask = ball_query(geom_points, centroids, self.radius, self.n_p)

        # ------------------------------------------------------------------
        # 4. Index global features + concat relative coords -> MLP -> max pool
        # ------------------------------------------------------------------
        # Expand global_feat for gathering: [B, N, d_model] -> [B, n_s, N, d_model]
        gf_exp = global_feat.unsqueeze(1).expand(B, effective_n_s, N, self.d_model)
        # Gather neighbour features: [B, n_s, n_p, d_model]
        grouped_feat = torch.gather(
            gf_exp,
            2,
            knn_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_model),
        )

        # Gather neighbour coordinates for relative offset: [B, n_s, n_p, d_in]
        gp_exp = geom_points.unsqueeze(1).expand(B, effective_n_s, N, d)
        grouped_xyz = torch.gather(
            gp_exp,
            2,
            knn_idx.unsqueeze(-1).expand(-1, -1, -1, d),
        )
        rel_xyz = grouped_xyz - centroids.unsqueeze(2)  # [B, n_s, n_p, d_in]

        # Concatenate: [B, n_s, n_p, d_model + d_in]
        group_input = torch.cat([grouped_feat, rel_xyz], dim=-1)

        # Flatten, apply MLP, reshape
        group_flat = group_input.reshape(B * effective_n_s * self.n_p, self.d_model + d)
        group_feat_out = self.group_mlp(group_flat)  # [B*n_s*n_p, d_model]
        group_feat_out = group_feat_out.view(B, effective_n_s, self.n_p, self.d_model)

        # Mask out invalid (padded) neighbours before max-pool
        knn_mask_exp = knn_mask.unsqueeze(-1)  # [B, n_s, n_p, 1]
        group_feat_out = group_feat_out.masked_fill(~knn_mask_exp, float("-inf"))

        # Max-pool over Np dimension -> [B, n_s, d_model]
        local_feat = group_feat_out.max(dim=2).values

        # Guard against all-invalid groups (shouldn't happen in practice)
        local_feat = torch.nan_to_num(local_feat, nan=0.0, posinf=0.0, neginf=0.0)

        # ------------------------------------------------------------------
        # 5. Cross-attention blocks: Q=local_feat, K=V=global_feat
        #    key_padding_mask: True for padded positions (format expected by
        #    nn.MultiheadAttention with key_padding_mask)
        # ------------------------------------------------------------------
        kv_pad_mask: Optional[torch.Tensor] = None
        if mask is not None:
            kv_pad_mask = ~mask  # [B, N] True = padded = should be ignored

        x = local_feat  # [B, n_s, d_model]
        for block in self.cross_attn_blocks:
            x = block(x, k=global_feat, v=global_feat, key_padding_mask=kv_pad_mask)

        # ------------------------------------------------------------------
        # 6. Self-attention blocks on the Ns centroid tokens
        #    When FPS is bypassed (effective_n_s == N), the centroid tokens
        #    include padded dummy points, so we must pass the mask to avoid
        #    contaminating the latent space.
        # ------------------------------------------------------------------
        sa_pad_mask = kv_pad_mask if effective_n_s == N else None
        for block in self.self_attn_blocks:
            x = block(x, key_padding_mask=sa_pad_mask)

        # ------------------------------------------------------------------
        # 7. Output projections -> encoder K and V
        # ------------------------------------------------------------------
        enc_k = self.k_out(x)  # [B, n_s, d_model]
        enc_v = self.v_out(x)  # [B, n_s, d_model]

        return enc_k, enc_v


class GINOTDecoder(nn.Module):
    """GINOT Solution Decoder.

    Decodes query point locations into solution values using the latent
    KEY and VALUE matrices produced by GINOTGeometryEncoder.

    Steps
    -----
    1. NeRF positional encoding + MLP -> Q matrix  [B, Nq, d_model]
    2. num_decoder_layers Cross-Attention blocks using encoder K and V
    3. Output MLP -> solution field  [B, Nq, d_out]
    """

    def __init__(
        self,
        d_in: int = 2,
        d_model: int = 128,
        d_out: int = 1,
        n_heads: int = 4,
        num_layers: int = 2,
        n_freqs: int = 6,
        mlp_hidden_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [d_model * 2, d_model * 2]

        self.d_model = d_model

        # NeRF positional encoding for query points
        self.posenc = FourierFeatures(n_freqs=n_freqs, scale=1.0, include_input=True)
        posenc_dim = self.posenc.out_dim

        # MLP: posenc_dim -> d_model  (query projection)
        self.query_mlp = MLP(
            in_dim=posenc_dim,
            hidden=mlp_hidden_dims,
            out_dim=d_model,
            act=nn.GELU,
            norm="layer",
        )

        # Cross-attention blocks
        self.cross_attn_blocks = nn.ModuleList(
            [_GINOTAttentionBlock(d_model, n_heads) for _ in range(num_layers)]
        )

        # Output MLP: d_model -> d_out
        self.out_mlp = MLP(
            in_dim=d_model,
            hidden=mlp_hidden_dims,
            out_dim=d_out,
            act=nn.GELU,
            norm="layer",
        )

    def forward(
        self,
        query_points: torch.Tensor,          # [B, Nq, d_in]
        enc_k: torch.Tensor,                 # [B, Ns, d_model]  encoder K
        enc_v: torch.Tensor,                 # [B, Ns, d_model]  encoder V
        query_mask: Optional[torch.Tensor] = None,  # [B, Nq] True=real
    ) -> torch.Tensor:
        """Returns [B, Nq, d_out]."""
        B, Nq, _ = query_points.shape

        # ------------------------------------------------------------------
        # 1. Query projection: posenc + MLP -> Q matrix
        # ------------------------------------------------------------------
        posenc_out = self.posenc(query_points)           # [B, Nq, posenc_dim]
        q_flat = posenc_out.reshape(B * Nq, -1)
        q = self.query_mlp(q_flat).view(B, Nq, self.d_model)  # [B, Nq, d_model]

        # ------------------------------------------------------------------
        # 2. Cross-attention blocks using encoder K and V
        #    Each block: (attn(Q, enc_k, enc_v) + residual) -> norm1 -> (FFN + residual) -> norm2
        # ------------------------------------------------------------------
        x = q
        for block in self.cross_attn_blocks:
            x = block(x, k=enc_k, v=enc_v)

        # ------------------------------------------------------------------
        # 3. Output MLP
        # ------------------------------------------------------------------
        x_flat = x.reshape(B * Nq, -1)
        out = self.out_mlp(x_flat).view(B, Nq, -1)  # [B, Nq, d_out]

        # Zero out padded query positions in the output
        if query_mask is not None:
            out = out * query_mask.unsqueeze(-1).to(out.dtype)

        return out


class GINOT(nn.Module):
    """Geometry-Informed Neural Operator with Transformers (GINOT).

    Combines GINOTGeometryEncoder and GINOTDecoder.

    Parameters
    ----------
    d_model : int
        Model/hidden dimension (used throughout encoder and decoder).
    num_encoder_cross_layers : int
        Number of cross-attention blocks in the geometry encoder.
    num_encoder_self_layers : int
        Number of self-attention blocks in the geometry encoder.
    num_decoder_layers : int
        Number of cross-attention blocks in the solution decoder.
    n_heads : int
        Number of attention heads.
    n_s : int or float
        Number of FPS centroids (encoder output tokens), or a float ratio
        in (0.0, 1.0] to dynamically compute the count per batch.
    n_p : int
        Maximum number of neighbours per centroid (ball query).
    radius : float
        Ball-query radius.
    mlp_hidden_dims : list of int, optional
        Hidden dimensions for all internal MLPs.  Defaults to
        [d_model*2, d_model*2] when None.
    d_in : int
        Spatial dimension of the input point cloud (default 2 for 2-D).
    d_out : int
        Output solution dimension (default 1 for scalar stress).
    n_freqs : int
        Number of NeRF positional encoding frequency bands.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_encoder_cross_layers: int = 2,
        num_encoder_self_layers: int = 2,
        num_decoder_layers: int = 2,
        n_heads: int = 4,
        n_s: Union[int, float] = 128,
        n_p: int = 32,
        radius: float = 0.1,
        mlp_hidden_dims: Optional[List[int]] = None,
        d_in: int = 2,
        d_out: int = 1,
        n_freqs: int = 6,
    ) -> None:
        super().__init__()
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [d_model * 2, d_model * 2]

        # Store all hyperparameters for get_arch()
        self._d_model = d_model
        self._num_encoder_cross_layers = num_encoder_cross_layers
        self._num_encoder_self_layers = num_encoder_self_layers
        self._num_decoder_layers = num_decoder_layers
        self._n_heads = n_heads
        self._n_s = n_s
        self._n_p = n_p
        self._radius = radius
        self._mlp_hidden_dims = list(mlp_hidden_dims)
        self._d_in = d_in
        self._d_out = d_out
        self._n_freqs = n_freqs

        self.encoder = GINOTGeometryEncoder(
            d_in=d_in,
            d_model=d_model,
            n_s=n_s,
            n_p=n_p,
            radius=radius,
            n_heads=n_heads,
            num_cross_layers=num_encoder_cross_layers,
            num_self_layers=num_encoder_self_layers,
            n_freqs=n_freqs,
            mlp_hidden_dims=mlp_hidden_dims,
        )

        self.decoder = GINOTDecoder(
            d_in=d_in,
            d_model=d_model,
            d_out=d_out,
            n_heads=n_heads,
            num_layers=num_decoder_layers,
            n_freqs=n_freqs,
            mlp_hidden_dims=mlp_hidden_dims,
        )

    def get_arch(self) -> dict:
        return {
            "d_model": self._d_model,
            "num_encoder_cross_layers": self._num_encoder_cross_layers,
            "num_encoder_self_layers": self._num_encoder_self_layers,
            "num_decoder_layers": self._num_decoder_layers,
            "n_heads": self._n_heads,
            "n_s": self._n_s,
            "n_p": self._n_p,
            "radius": self._radius,
            "mlp_hidden_dims": self._mlp_hidden_dims,
            "d_in": self._d_in,
            "d_out": self._d_out,
            "n_freqs": self._n_freqs,
        }

    def forward(
        self,
        geom_points: torch.Tensor,           # [B, N, d_in]
        query_points: torch.Tensor,          # [B, Nq, d_in]
        mask: Optional[torch.Tensor] = None, # [B, N] bool, True=real point
    ) -> torch.Tensor:
        """Predict solution field at query locations.

        Parameters
        ----------
        geom_points : Tensor [B, N, d_in]
            Boundary point cloud, possibly zero-padded.
        query_points : Tensor [B, Nq, d_in]
            Query locations.
        mask : BoolTensor [B, N], optional
            True for real geometry points, False for zero-padded dummy points.
            When geom_points and query_points have the same sequence length
            (e.g. batched_all training mode), the mask is also applied to the
            decoder output to zero out padded query positions.

        Returns
        -------
        Tensor [B, Nq, d_out]
        """
        # Encode geometry
        enc_k, enc_v = self.encoder(geom_points, mask=mask)  # [B, Ns, d_model] each

        # Determine query mask: only applicable when geom and query are co-padded
        # (same sequence length), which is always the case in batched_all training.
        query_mask: Optional[torch.Tensor] = None
        if mask is not None and geom_points.shape[1] == query_points.shape[1]:
            query_mask = mask

        # Decode at query locations
        out = self.decoder(query_points, enc_k, enc_v, query_mask=query_mask)  # [B, Nq, d_out]

        return out
