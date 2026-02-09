import os
import sys
import time
import torch
from pathlib import Path

# # Pin this process to GPU 0 BEFORE importing training_script (which selects device at import time)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Ensure directory is on path for import resolution
this_dir = Path(__file__).parent.resolve()
if str(this_dir) not in sys.path:
    sys.path.insert(0, str(this_dir))

from Training_script import main as train_main  # noqa: E402


PRESETS_GPU0 = [
    "S0",
    "S",
    "S_full",
    "S_full_ln_do",
    "S_ln_maxmean",
    "S_gn8_maxmean",
    "S+",
]


def run_with_fallback(preset: str, initial_batch: int) -> bool:
    """Try training with a sequence of decreasing batch sizes upon OOM."""
    batch_plan = [initial_batch, max(1, initial_batch // 2), 1]
    for b in batch_plan:
        try:
            print(f"\n[GPU0] Preset={preset} | Trying batch={b}")
            train_main(preset, b)
            print(f"[GPU0] Preset={preset} | Completed with batch={b}")
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
            print(f"[GPU0] Non-recoverable RuntimeError for preset {preset}: {e}")
            return False
        except Exception as e:
            print(f"[GPU0] Unexpected error for preset {preset}: {e}")
            return False
    return False


def main_gpu0() -> None:
    print("Starting GPU0 preset run set...")
    for preset in PRESETS_GPU0:
        # Small presets can start higher
        run_with_fallback(preset, initial_batch=40)
    print("GPU0 run set finished.")


if __name__ == "__main__":
    main_gpu0()
