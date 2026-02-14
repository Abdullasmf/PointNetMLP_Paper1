import torch
import torch.nn as nn
import math
from typing import List, Dict, Any, Optional

# Import necessary components from pn_models
from pn_models import PointNet2Encoder2D, MLP

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
