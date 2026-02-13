import os
import sys
import time
import torch
from pathlib import Path



# Ensure directory is on path for import resolution
this_dir = Path(__file__).parent.resolve()
if str(this_dir) not in sys.path:
    sys.path.insert(0, str(this_dir))

from Training_script import main as train_main  # noqa: E402


PRESETS_GPU0 = [
    ["L","L_bracket"],
    ["L","Plate_hole"],
]


def run_with_fallback(preset: str, initial_batch: int, geometry: str) -> bool:
    """Try training with a sequence of decreasing batch sizes upon OOM."""
    iterative = int(initial_batch *0.1)
    batch_plan = list(range(initial_batch, 0, -iterative))
    batch_plan.append(1)
    for b in batch_plan:
        try:
            print(f"\n[GPU0] Preset={preset} | Trying batch={b} | Geometry={geometry}")
            train_main(preset, b, geometry)
            print(f"[GPU0] Preset={preset} | Completed with batch={b} | Geometry={geometry}")
            return True
        except RuntimeError as e:
            low = str(e).lower()
            if ("out of memory" in low or "cuda" in low) and b != 1:
                print(f"[GPU0] OOM/CUDA at batch {b}; reducing and retrying...")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                time.sleep(2)
                continue
            print(f"[GPU0] Non-recoverable RuntimeError for preset {preset} | Geometry={geometry}: {e}")
            return False
        except Exception as e:
            print(f"[GPU0] Unexpected error for preset {preset} | Geometry={geometry}: {e}")
            return False
    return False


def main_gpu0() -> None:
    print("Starting GPU0 preset run set...")
    project_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = Path(project_dir).parent #parent folder name is model name
    model_name = parent_dir.name
    print(model_name)
    for preset in PRESETS_GPU0:
        # Small presets can start higher
        run_with_fallback(preset[0], initial_batch=200, geometry=preset[1])
    print("GPU0 run set finished.")


if __name__ == "__main__":
    main_gpu0()
