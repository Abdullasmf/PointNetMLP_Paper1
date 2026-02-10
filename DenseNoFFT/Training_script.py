import random
import json
import hashlib
from typing import List, Tuple, Dict, Optional

import h5py
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from pn_models import PointNetMLPJoint
from benchmarks import DenseNoFFT

project_dir = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)

# Defer device prints and data loading to main() to avoid re-exec in worker processes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_h5_pointsets(path: Path) -> List[torch.Tensor]:
    all_data = []
    coord_stress_list = []
    with h5py.File(path, 'r') as hf:
        # Sort keys to ensure numerical order (sample_0, sample_1, ...)
        # splitting by '_' and taking the last part ensures 'sample_10' comes after 'sample_2'
        keys = sorted(hf.keys(), key=lambda x: int(x.split('_')[1]))
        
        print(f"Found {len(keys)} samples. Loading...")
        
        for key in keys:
            group = hf[key]
            
            # Create a dictionary for this sample
            # Handle both 'corner' (L_bracket) and 'params' (Plate_hole) metadata
            sample = {
                'points': group['points'][:],  # (N, 2)
                'stress': group['stress'][:],  # (N, 1)
            }
            # Add metadata if present (not used in training, but kept for compatibility)
            if 'corner' in group:
                sample['corner'] = group['corner'][:]
            if 'params' in group:
                sample['params'] = group['params'][:]
            
            coord_stress = np.hstack((sample['points'], sample['stress']))  # (N, 3)
            coord_stress_list.append(torch.from_numpy(coord_stress).float())
            all_data.append(sample)
        
    return coord_stress_list


# Loaded in main()


class GeomStressDataset(Dataset):
    """
    Geometry-level dataset. Each item is one simulation geometry with variable number of points.
    x: [N,2] coordinates, y: [N,1] stress.
    Applies global normalization using provided stats.
    """

    def __init__(
        self,
        tensors: List[torch.Tensor],
        coord_center: torch.Tensor,
        coord_half_range: torch.Tensor,
        stress_mean: torch.Tensor,
        stress_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.items: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.coord_center = coord_center
        self.coord_half_range = torch.clamp(coord_half_range, min=1e-8)
        self.stress_mean = stress_mean
        self.stress_std = torch.clamp(stress_std, min=1e-8)
        for t in tensors:
            if t.shape[1] < 3:
                raise ValueError("Each tensor must be [N,3]: x,y,stress")
            xy = t[:, :2].contiguous()
            s = t[:, 2:3].contiguous()
            self.items.append((xy, s))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        xy, s = self.items[idx]
        # Normalize with GLOBAL stats computed from training set
        xyn = (xy - self.coord_center) / self.coord_half_range
        sn = (s - self.stress_mean) / self.stress_std
        return {
            "points": xyn,  # [N,2]
            "stress": sn,  # [N,1]
            # Provide also unnormalized for potential analysis if needed
            "points_raw": xy,
            "stress_raw": s,
        }


def default_collate_variable(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    # We expect batch_size=1; keep interface flexible
    assert len(batch) == 1, "Use batch_size=1 for variable-size point sets."
    return batch[0]


def make_collate_fixed_points(k: int):
    """Return a collate function that samples exactly k points per geometry (with replacement if N<k)
    and stacks a batch: points [B,k,2], stress [B,k,1].
    """

    def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pts_b: List[torch.Tensor] = []
        s_b: List[torch.Tensor] = []
        for item in batch:
            pts = item["points"]  # [N,2] on CPU
            s = item["stress"]  # [N,1]
            N = pts.shape[0]
            if N >= k:
                idx = torch.randperm(N)[:k]
            else:
                # sample with replacement to reach k
                idx = torch.randint(0, N, (k,))
            pts_b.append(pts[idx])
            s_b.append(s[idx])
        points = torch.stack(pts_b, dim=0)  # [B,k,2]
        stress = torch.stack(s_b, dim=0)  # [B,k,1]
        return {"points": points, "stress": stress}

    return _collate


class DualSamplerCollate:
    """Pickle-safe callable collate that samples two point sets per geometry.

    Returns dict with keys:
      - 'geom_points': [B,K_enc,2]
      - 'query_points': [B,K_q,2]
      - 'stress': [B,K_q,1]
    """

    def __init__(self, k_enc: int, k_q: int) -> None:
        self.k_enc = int(k_enc)
        self.k_q = int(k_q)

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        gp_b: List[torch.Tensor] = []
        qp_b: List[torch.Tensor] = []
        s_b: List[torch.Tensor] = []
        for item in batch:
            pts = item["points"]  # [N,2] on CPU
            s = item["stress"]  # [N,1]
            N = pts.shape[0]
            # Encoder samples
            if N >= self.k_enc:
                idx_enc = torch.randperm(N)[: self.k_enc]
            else:
                idx_enc = torch.randint(0, N, (self.k_enc,))
            # Query samples
            if N >= self.k_q:
                idx_q = torch.randperm(N)[: self.k_q]
            else:
                idx_q = torch.randint(0, N, (self.k_q,))
            gp_b.append(pts[idx_enc])
            qp_b.append(pts[idx_q])
            s_b.append(s[idx_q])
        return {
            "geom_points": torch.stack(gp_b, dim=0),
            "query_points": torch.stack(qp_b, dim=0),
            "stress": torch.stack(s_b, dim=0),
        }


class AllNodesPadCollate:
    """Pickle-safe collate that uses ALL nodes per geometry.

    Pads each geometry in the batch to the maximum node count by repeating valid indices
    (no zero-padding), and returns a dual-set dict compatible with the model forward:
      - 'geom_points': [B,maxN,2]
      - 'query_points': [B,maxN,2]
      - 'stress': [B,maxN,1]
    """

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        Ns = [item["points"].shape[0] for item in batch]
        maxN = max(Ns)
        gp_b: List[torch.Tensor] = []
        qp_b: List[torch.Tensor] = []
        s_b: List[torch.Tensor] = []
        orig_idx_b: List[torch.Tensor] = []
        for item, N in zip(batch, Ns):
            pts = item["points"]  # [N,2]
            s = item["stress"]  # [N,1]
            idx_all = torch.arange(N)
            if N < maxN:
                extra = torch.randint(0, N, (maxN - N,))
                enc_idx = torch.cat([idx_all, extra], dim=0)
                qry_idx = enc_idx
            else:
                enc_idx = idx_all
                qry_idx = idx_all
            gp_b.append(pts[enc_idx])
            qp_b.append(pts[qry_idx])
            s_b.append(s[qry_idx])
            # Track original index per position for weighting duplicates
            orig_idx_b.append(qry_idx)
        return {
            "geom_points": torch.stack(gp_b, dim=0),
            "query_points": torch.stack(qp_b, dim=0),
            "stress": torch.stack(s_b, dim=0),
            "orig_idx": torch.stack(orig_idx_b, dim=0),  # [B,maxN]
        }


def compute_global_normalization(
    train_tensors: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute coord center/half-range and stress mean/std from training set."""
    all_xy = torch.cat([t[:, :2] for t in train_tensors], dim=0)
    all_s = torch.cat([t[:, 2:3] for t in train_tensors], dim=0)
    xy_min = all_xy.min(dim=0).values
    xy_max = all_xy.max(dim=0).values
    coord_center = 0.5 * (xy_min + xy_max)
    coord_half_range = torch.clamp(0.5 * (xy_max - xy_min), min=1e-6)
    stress_mean = all_s.mean(dim=0)
    stress_std = all_s.std(dim=0, unbiased=False).clamp(min=1e-6)
    return coord_center, coord_half_range, stress_mean, stress_std


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    coord_center: torch.Tensor,
    coord_half_range: torch.Tensor,
    stress_mean: torch.Tensor,
    stress_std: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_points_per_geom: Optional[int] = None,
    grad_clip_norm: Optional[float] = 1.0,
    save_path: Optional[Path] = None,
    early_stopping_patience: Optional[int] = 20,
    early_stopping_min_delta: float = 0.0,
    use_amp: bool = False,
) -> None:
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_amp))
    best_val = float("inf")
    # For MPa-space validation logging
    stress_mean_d = stress_mean.to(device)
    stress_std_d = stress_std.to(device)

    # Learning rate schedule (optional OneCycleLR based on rough steps)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=10.0,
        anneal_strategy="cos",
        three_phase=False,
    )

    mse = nn.MSELoss()
    t0 = time.time()
    epochs_since_improve = 0
    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        model.train()
        train_loss = 0.0
        ntrain = 0
        for batch in train_loader:
            # default for duplicate-aware weighting (used in batched_all)
            weights: Optional[torch.Tensor] = None
            if "geom_points" in batch:
                # Dual-sampling batched mode
                gp: torch.Tensor = batch["geom_points"].to(device)  # [B,Kenc,2]
                query_xy: torch.Tensor = batch["query_points"].to(device)  # [B,Kq,2]
                target: torch.Tensor = batch["stress"].to(device)  # [B,Kq,1]
                B, Kq, _ = query_xy.shape
                Bmul = B * Kq
                # Optional sample weights for batched_all to down-weight duplicates
                if "orig_idx" in batch:
                    orig_idx = batch["orig_idx"]  # [B,Kq] on CPU
                    # Build weights inversely proportional to replication counts per geometry
                    ws = []
                    for b in range(orig_idx.shape[0]):
                        idx_b = orig_idx[b]
                        # counts per original index value
                        max_idx = int(idx_b.max().item()) + 1
                        counts = torch.bincount(idx_b, minlength=max_idx)
                        counts[counts == 0] = 1  # avoid div by zero
                        w_b = 1.0 / counts[idx_b]
                        # normalize per-geometry to mean 1
                        w_b = w_b / w_b.mean()
                        ws.append(w_b)
                    weights = (
                        torch.stack(ws, dim=0).to(device).unsqueeze(-1)
                    )  # [B,Kq,1]
            else:
                # Full-geometry single/batched mode from earlier
                pts: torch.Tensor = batch["points"].to(device)  # [N,2] or [B,K,2]
                stress: torch.Tensor = batch["stress"].to(device)  # [N,1] or [B,K,1]
                if pts.dim() == 2:
                    # Single geometry
                    N = pts.shape[0]
                    if max_points_per_geom is None:
                        query_xy = pts
                        target = stress
                    else:
                        q = min(N, max_points_per_geom)
                        idxs = torch.randperm(N, device=device)[:q]
                        query_xy = pts[idxs]
                        target = stress[idxs]
                    gp = pts.unsqueeze(0)  # [1,N,2]
                    query_xy = query_xy.unsqueeze(0)  # [1,q,2] or [1,N,2]
                    target = target.unsqueeze(0)
                    Bmul = query_xy.shape[1]
                else:
                    # Batched [B,K,*]
                    B, K, _ = pts.shape
                    if max_points_per_geom is None or max_points_per_geom >= K:
                        query_xy = pts
                        target = stress
                        Bmul = B * K
                    else:
                        q = max_points_per_geom
                        # Build per-batch indices
                        idxs = torch.stack(
                            [torch.randperm(K, device=device)[:q] for _ in range(B)],
                            dim=0,
                        )  # [B,q]
                        bidx = (
                            torch.arange(B, device=device).unsqueeze(-1).expand(-1, q)
                        )
                        query_xy = pts[bidx, idxs, :]
                        target = stress[bidx, idxs, :]
                        Bmul = B * q
                    gp = pts  # use same points for encoder

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=(device.type == "cuda" and use_amp)
            ):
                pred = model(gp, query_xy)  # [B,Kq,1]
                # Use sample weights if provided (batched_all duplicate adjustment). Guard with 'is not None'.
                if isinstance(weights, torch.Tensor):
                    diff2 = (pred - target) ** 2
                    loss = (diff2 * weights).mean()
                else:
                    loss = mse(pred, target)

            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item() * Bmul
            ntrain += Bmul

        train_loss /= max(1, ntrain)

        # Validation
        model.eval()
        val_loss = 0.0
        nval = 0
        val_loss_mpa_sum = 0.0
        # For R2 in MPa-space
        ss_res_mpa = 0.0
        sum_y_mpa = 0.0
        sum_y2_mpa = 0.0
        count_mpa = 0
        with torch.no_grad():
            for batch in val_loader:
                pts: torch.Tensor = batch["points"].to(device)  # [N,2]
                stress: torch.Tensor = batch["stress"].to(device)  # [N,1]
                N = pts.shape[0]
                with torch.amp.autocast(
                    "cuda", enabled=(device.type == "cuda" and use_amp)
                ):
                    pred = model(pts.unsqueeze(0), pts.unsqueeze(0)).squeeze(0)  # [N,1]
                loss = mse(pred, stress)
                val_loss += loss.item() * N
                # De-normalized MSE in MPa
                pred_mpa = pred * stress_std_d + stress_mean_d
                stress_mpa = stress * stress_std_d + stress_mean_d
                diff = pred_mpa - stress_mpa
                val_loss_mpa_sum += torch.sum(diff**2).item()
                # Accumulate for R2
                ss_res_mpa += torch.sum(diff**2).item()
                sum_y_mpa += torch.sum(stress_mpa).item()
                sum_y2_mpa += torch.sum(stress_mpa**2).item()
                count_mpa += int(N)
                nval += N
        val_loss /= max(1, nval)
        val_loss_mpa = val_loss_mpa_sum / max(1, nval)
        # Compute R2 in MPa space
        if count_mpa > 0:
            mean_y = sum_y_mpa / count_mpa
            ss_tot = max(1e-12, sum_y2_mpa - count_mpa * (mean_y**2))
            r2_mpa = 1.0 - (ss_res_mpa / ss_tot)
        else:
            r2_mpa = float("nan")
        epoch_dt = time.time() - epoch_t0

        print(
            f"Epoch {epoch:03d} | train MSE: {train_loss:.6f} | val MSE: {val_loss:.6f} | val MSE(MPa^2): {val_loss_mpa:.3f} | R2(MPa): {r2_mpa:.4f} | lr: {scheduler.get_last_lr()[0]:.2e} | epoch: {epoch_dt:.1f}s"
        )

        # Checkpoint best
        if val_loss < (best_val - early_stopping_min_delta):
            best_val = val_loss
            epochs_since_improve = 0
            if save_path is not None:
                ckpt = {
                    "model_state": model.state_dict(),
                    "arch": model.get_arch() if hasattr(model, "get_arch") else None,
                    "coord_center": coord_center.cpu(),
                    "coord_half_range": coord_half_range.cpu(),
                    "stress_mean": stress_mean.cpu(),
                    "stress_std": stress_std.cpu(),
                    "config": {
                        "epochs_trained": epoch,
                        "best_val": best_val,
                    },
                }
                # Convenience for validators expecting this key name
                ckpt["best_val_loss"] = best_val
                torch.save(ckpt, str(save_path))
                print(f"Saved best model to: {save_path}")
        else:
            epochs_since_improve += 1
            if (
                early_stopping_patience is not None
                and epochs_since_improve >= early_stopping_patience
            ):
                print(
                    f"Early stopping triggered after {epochs_since_improve} epochs without improvement. Best val MSE: {best_val:.6f}"
                )
                break

    dt = time.time() - t0
    print(f"Training finished in {dt/60:.1f} min. Best val MSE: {best_val:.6f}")


def main(preset_name: str = "S0", batch=8, dataset: str = "L_bracket") -> None:
    # preset_name = "S0"
    # batch = 8
    # dataset = "L_bracket" or "Plate_hole"
    print(
        f"Starting training script with preset '{preset_name}', batch size {batch}, and dataset '{dataset}'"
    )
    # Device/backend setup

    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    print(f"Using device: {device}")

    # Locate HDF5 file based on dataset choice
    # Ensure we look in the appropriate folder for the stress data
    parent_dir = Path(project_dir).parent
    if dataset == "L_bracket":
        h5py_path = Path(parent_dir, "L_Bracket", "L_bracket_stress.h5")
        geom_prefix = "L-"
    elif dataset == "Plate_hole":
        h5py_path = Path(parent_dir, "Plate_Hole", "Plate_hole_stress.h5")
        geom_prefix = "H-"
    else:
        raise ValueError(f"Invalid dataset '{dataset}'. Must be 'L_bracket' or 'Plate_hole'")

    print(f"Loading data from: {h5py_path}")
    PS_list_whole = load_h5_pointsets(h5py_path)
    print(f"Loaded {len(PS_list_whole)} datasets from the HDF5 file.")

    # Load external presets JSON to allow expanding model zoo without editing this script
    presets_path = Path(project_dir, "model_presets.json")
    if not presets_path.exists():
        raise FileNotFoundError(
            f"Preset file 'model_presets.json' not found at {presets_path}. Please create it or copy the provided template."
        )
    with open(presets_path, "r", encoding="utf-8") as f:
        try:
            PRESETS = json.load(f)
        except Exception as exc:
            raise RuntimeError(
                "Failed to parse model_presets.json (invalid JSON)"
            ) from exc
    if preset_name not in PRESETS:
        raise KeyError(
            f"Preset '{preset_name}' not found. Available presets: {', '.join(sorted(PRESETS.keys()))}"
        )

    _cfg = PRESETS[preset_name]
    # In-file configuration (no CLI needed)
    epochs: int = 500
    lr: float = 3e-4
    weight_decay: float = 1e-4
    # Use all points every step (no subsampling of queries)
    max_points_per_geom: Optional[int] = (
        None  # set to an int to sample per-geometry points per step
    )
    # Early stopping
    early_stopping_patience: int = 40
    early_stopping_min_delta: float = 0.0
    # Architecture
    latent_dim: int = int(_cfg["latent_dim"])  # encoder latent size
    pre_hidden: List[int] = list(_cfg["pre_hidden"])  # pre-MLP on coords
    sa_blocks: List[dict] = list(_cfg["sa_blocks"])  # set abstraction blocks
    gf_hidden: List[int] = list(_cfg["gf_hidden"])  # global feature head
    head_hidden: List[int] = list(_cfg["head_hidden"])  # MLP head sizes
    # Optional human-readable model name (prefix for the file); set to None to use default
    model_name: Optional[str] = None  # e.g., "pn_small_r0p08"

    # Fourier positional encodings to enhance spatial/detail sensitivity
    # Allow overriding positional encodings per preset; default to 4 freqs if unspecified
    posenc = _cfg.get("posenc", {"n_freqs": 4, "scale": 1.0})
    head_posenc = _cfg.get("head_posenc", {"n_freqs": 4, "scale": 1.0})

    # Normalization/pooling flags (encoder + head). Defaults keep backward compatibility
    enc_norm: str = str(_cfg.get("norm", "batch"))
    enc_num_groups: int = int(_cfg.get("num_groups", 16))
    enc_pool: str = str(_cfg.get("pool", "max"))  # 'max' | 'max+mean'
    head_norm: str = str(_cfg.get("head_norm", "batch"))
    head_dropout: float = float(_cfg.get("head_dropout", 0.0))

    # Save path (unique per-architecture; overwrites across runs for the same arch)
    arch_for_hash = {
        "latent_dim": latent_dim,
        "pre_hidden": pre_hidden,
        "sa_blocks": sa_blocks,
        "gf_hidden": gf_hidden,
        "head_hidden": head_hidden,
        "posenc": posenc,
        "head_posenc": head_posenc,
        "norm": enc_norm,
        "num_groups": enc_num_groups,
        "pool": enc_pool,
        "head_norm": head_norm,
        "head_dropout": head_dropout,
    }
    arch_hash = hashlib.md5(
        json.dumps(arch_for_hash, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    save_dir = Path(project_dir, "Trained_models")
    base_name = model_name if model_name else "pnmlp"
    save_path = save_dir / f"{geom_prefix}{base_name}_{arch_hash}.pt"

    set_seed(42)

    # Split geometries into train/val
    n_geoms = len(PS_list_whole)
    idxs = list(range(n_geoms))
    train_idx, val_idx = train_test_split(idxs, test_size=0.2, random_state=42)

    train_tensors = [PS_list_whole[i] for i in train_idx]
    val_tensors = [PS_list_whole[i] for i in val_idx]

    coord_center, coord_half_range, stress_mean, stress_std = (
        compute_global_normalization(train_tensors)
    )
    print(
        f"Coord center={coord_center.numpy()}, half_range={coord_half_range.numpy()} | stress_mean={stress_mean.item():.4f}, stress_std={stress_std.item():.4f}"
    )

    train_ds = GeomStressDataset(
        train_tensors, coord_center, coord_half_range, stress_mean, stress_std
    )
    val_ds = GeomStressDataset(
        val_tensors, coord_center, coord_half_range, stress_mean, stress_std
    )

    # Training mode: "full" (encoder sees ALL points, batch_size=1),
    #                "batched" (dual sampling), or
    #                "batched_all" (all nodes per geometry with padded repeats to batch max)
    train_mode = "batched_all"  # default per request
    if train_mode == "full":
        train_batch_size: int = 1
        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=default_collate_variable,
            pin_memory=(device.type == "cuda"),
            persistent_workers=False,
        )
        # keep model forward using all points
        max_points_per_geom: Optional[int] = None
    elif train_mode == "batched":
        # Dual-sampling high-utilization settings (tune to your VRAM)
        train_batch_size: int = 4
        k_enc: int = 8192  # encoder points per geometry
        k_q: int = 8192  # query points per geometry
        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=DualSamplerCollate(k_enc, k_q),
            pin_memory=(device.type == "cuda"),
            persistent_workers=True,
        )
        # no per-geometry query subsampling in the loop (dual provides fixed K)
        max_points_per_geom = None
    else:
        # batched_all: use ALL nodes per geometry; pad to batch max by repeating real indices (no zero pads)
        train_batch_size: int = batch
        print(f"Using 'batched_all' training with batch size {train_batch_size}")
        # Use top-level AllNodesPadCollate (pickle-safe). No local redefinition.
        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=AllNodesPadCollate(),
            pin_memory=(device.type == "cuda"),
            persistent_workers=True,
        )
        max_points_per_geom = None
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=default_collate_variable,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    # Build architecture config from flags
    encoder_cfg = {
        "latent_dim": latent_dim,
        "pre_hidden": pre_hidden,
        "sa_blocks": sa_blocks,
        "gf_hidden": gf_hidden,
        "posenc": posenc,
        "head_posenc": head_posenc,
        # new flags
        "norm": enc_norm,
        "num_groups": enc_num_groups,
        "pool": enc_pool,
        "head_norm": head_norm,
        "head_dropout": head_dropout,
    }

    model = DenseNoFFT(
        latent_dim=latent_dim, mlp_hidden=head_hidden, encoder_cfg=encoder_cfg
    )

    # Ensure save directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving best checkpoint to: {save_path}")
    train(
        model,
        train_loader,
        val_loader,
        coord_center,
        coord_half_range,
        stress_mean,
        stress_std,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        max_points_per_geom=max_points_per_geom,
        grad_clip_norm=0.5,
        save_path=save_path,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        use_amp=(device.type == "cuda"),
    )


if __name__ == "__main__":
    try:
        # Choose dataset: "L_bracket" for L-bracket geometry or "Plate_hole" for hole plate geometry
        main("L", batch=8, dataset="L_bracket")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
