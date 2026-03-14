"""Standalone script to compute the Signed Distance Function (SDF) for
2D FEM geometries (L-bracket and Plate with hole) and append it as
an input feature to the existing HDF5 datasets.

The SDF value for each mesh node is its minimum distance to the nearest
boundary edge of the domain (positive inside, zero on the boundary).

Usage:
    python compute_sdf.py --dataset L_bracket
    python compute_sdf.py --dataset Plate_hole
    python compute_sdf.py --dataset both    (default)
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Low-level geometry helpers
# ---------------------------------------------------------------------------

def _dist_point_to_segment_batch(
    px: np.ndarray, py: np.ndarray,
    ax: float, ay: float, bx: float, by: float,
) -> np.ndarray:
    """Minimum distance from each point (px[i], py[i]) to the segment (ax,ay)-(bx,by).

    Fully vectorised; px/py are 1-D arrays of length N.
    """
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    if len2 < 1e-12:
        return np.hypot(px - ax, py - ay)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / len2, 0.0, 1.0)
    cx_ = ax + t * dx
    cy_ = ay + t * dy
    return np.hypot(px - cx_, py - cy_)


# ---------------------------------------------------------------------------
# Per-geometry SDF computation
# ---------------------------------------------------------------------------

def compute_sdf_l_bracket(points: np.ndarray, corner: np.ndarray) -> np.ndarray:
    """Compute SDF for the filleted L-bracket domain (fillet ignored).

    Supports the expanded corner format:
      corner[0:2] = [xc, yc]  – absolute re-entrant corner coordinates
      corner[2:7] = [W, H, x_offset, y_offset, fillet_radius]  (new, optional)

    For old datasets with only 2 values, falls back to unit-square boundaries.

    Boundary segments (counter-clockwise):
      1. Bottom       : (x_offset, y_offset) -> (x_offset+W, y_offset)
      2. Right-lower  : (x_offset+W, y_offset) -> (x_offset+W, yc)
      3. Shelf        : (x_offset+W, yc) -> (xc, yc)   [inner horizontal]
      4. Inner-vert   : (xc, yc) -> (xc, y_offset+H)   [inner vertical]
      5. Top          : (xc, y_offset+H) -> (x_offset, y_offset+H)
      6. Left         : (x_offset, y_offset+H) -> (x_offset, y_offset)

    Parameters
    ----------
    points : (N, 2) – mesh node coordinates
    corner : [xc, yc] or [xc, yc, W, H, x_offset, y_offset, fillet_radius]

    Returns
    -------
    sdf : (N,) – distance to nearest boundary edge (>= 0 for interior nodes)
    """
    xc, yc = float(corner[0]), float(corner[1])
    if len(corner) >= 7:
        W, H = float(corner[2]), float(corner[3])
        x_offset, y_offset = float(corner[4]), float(corner[5])
    else:
        # Backward compatibility: assume unit square
        W, H, x_offset, y_offset = 1.0, 1.0, 0.0, 0.0
    segments = [
        (x_offset,      y_offset,      x_offset + W,  y_offset),      # bottom
        (x_offset + W,  y_offset,      x_offset + W,  yc),            # right-lower
        (x_offset + W,  yc,            xc,             yc),            # shelf
        (xc,            yc,            xc,             y_offset + H),  # inner vertical
        (xc,            y_offset + H,  x_offset,       y_offset + H),  # top
        (x_offset,      y_offset + H,  x_offset,       y_offset),      # left
    ]
    px, py = points[:, 0], points[:, 1]
    dists = np.stack(
        [_dist_point_to_segment_batch(px, py, ax, ay, bx, by)
         for ax, ay, bx, by in segments],
        axis=-1,
    )  # (N, 6)
    return dists.min(axis=-1)  # (N,)


def compute_sdf_plate_hole(points: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Compute SDF for the plate-with-hole domain.

    Supports the expanded params format:
      params[0:3] = [cx, cy, r]  – hole centre and radius
      params[3:7] = [W, H, x_offset, y_offset]  (new, optional)

    For old datasets with only 3 values, falls back to the unit-square outer
    boundary.

    For each interior node the SDF equals the minimum of:
      - distance to the outer rectangular boundary, and
      - distance to the circular hole boundary.

    Parameters
    ----------
    points : (N, 2) – mesh node coordinates
    params : [cx, cy, r] or [cx, cy, r, W, H, x_offset, y_offset]

    Returns
    -------
    sdf : (N,) – distance to nearest boundary (>= 0 for interior nodes)
    """
    cx, cy, r = float(params[0]), float(params[1]), float(params[2])
    if len(params) >= 7:
        W, H = float(params[3]), float(params[4])
        x_offset, y_offset = float(params[5]), float(params[6])
    else:
        # Backward compatibility: assume unit square
        W, H, x_offset, y_offset = 1.0, 1.0, 0.0, 0.0
    px, py = points[:, 0], points[:, 1]

    # Distance to the four outer edges of the bounding box
    dist_outer = np.minimum.reduce([
        px - x_offset,
        (x_offset + W) - px,
        py - y_offset,
        (y_offset + H) - py,
    ])  # (N,)

    # Distance to the circular hole boundary
    dist_hole = np.abs(np.hypot(px - cx, py - cy) - r)  # (N,)

    return np.minimum(dist_outer, dist_hole)  # (N,)


# ---------------------------------------------------------------------------
# HDF5 processing
# ---------------------------------------------------------------------------

def add_sdf_to_h5(h5_path: Path, geometry: str) -> None:
    """Open an HDF5 file in read-write mode and add 'sdf' datasets to every
    sample group that does not already have one.

    Parameters
    ----------
    h5_path  : path to the HDF5 file
    geometry : 'L_bracket' or 'Plate_hole'
    """
    print(f"Processing {h5_path} ...")
    with h5py.File(h5_path, "a") as hf:
        keys = sorted(hf.keys(), key=lambda k: int(k.split("_")[1]))
        n_added = 0
        for key in keys:
            grp = hf[key]
            if "sdf" in grp:
                continue  # already computed; skip

            points = grp["points"][:]  # (N, 2)

            if geometry == "L_bracket":
                if "corner" not in grp:
                    raise KeyError(
                        f"Sample '{key}' is missing the 'corner' dataset."
                    )
                sdf = compute_sdf_l_bracket(points, grp["corner"][:])
            elif geometry == "Plate_hole":
                if "params" not in grp:
                    raise KeyError(
                        f"Sample '{key}' is missing the 'params' dataset."
                    )
                sdf = compute_sdf_plate_hole(points, grp["params"][:])
            else:
                raise ValueError(
                    f"Unknown geometry '{geometry}'. "
                    "Choose 'L_bracket' or 'Plate_hole'."
                )

            grp.create_dataset(
                "sdf", data=sdf.reshape(-1, 1).astype(np.float32)
            )
            n_added += 1
            if n_added % 500 == 0:
                print(f"  Added SDF to {n_added}/{len(keys)} samples …")

    print(f"  Finished – {n_added} SDF datasets written to {h5_path.name}.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute SDF features for HDF5 geometry datasets.",
    )
    parser.add_argument(
        "--dataset",
        choices=["L_bracket", "Plate_hole", "both"],
        default="both",
        help="Which dataset to process (default: both).",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).parent.resolve()
    # parent_dir = project_dir.parent
    parent_dir = project_dir
    targets = []
    if args.dataset in ("L_bracket", "both"):
        targets.append(
            (parent_dir / "L_Bracket" / "L_bracket_stress.h5", "L_bracket")
        )
    if args.dataset in ("Plate_hole", "both"):
        targets.append(
            (parent_dir / "Plate_Hole" / "Plate_hole_stress.h5", "Plate_hole")
        )

    for h5_path, geometry in targets:
        if not h5_path.exists():
            print(f"Warning: {h5_path} not found; skipping.")
            continue
        add_sdf_to_h5(h5_path, geometry)


if __name__ == "__main__":
    main()
