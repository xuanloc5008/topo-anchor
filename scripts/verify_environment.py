from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


REQUIRED = [
    ("torch", "PyTorch training"),
    ("lightning", "training loop"),
    ("monai", "medical imaging transforms"),
    ("nibabel", "NIfTI I/O"),
    ("skimage", "mask topology descriptors"),
    ("gudhi", "cubical persistence"),
    ("gtda", "persistence vectorization"),
]

OPTIONAL_ACCELERATED = [
    ("mamba_ssm", "Res-Mamba blocks; required for paper training with model.mamba.backend=mamba_ssm"),
]


def main() -> int:
    failures: list[str] = []
    for module, purpose in REQUIRED:
        ok = importlib.util.find_spec(module) is not None
        status = "OK" if ok else "MISSING"
        print(f"{status:8s} {module:12s} {purpose}")
        if not ok:
            failures.append(module)

    optional_missing = []
    for module, purpose in OPTIONAL_ACCELERATED:
        ok = importlib.util.find_spec(module) is not None
        status = "OK" if ok else "MISSING"
        print(f"{status:8s} {module:12s} {purpose}")
        if not ok:
            optional_missing.append(module)

    if importlib.util.find_spec("torch") is not None:
        import torch

        print(f"torch_version={torch.__version__}")
        print(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device_count={torch.cuda.device_count()}")
            print(f"cuda_device_name={torch.cuda.get_device_name(0)}")

    if failures:
        print("\nMissing required dependencies: " + ", ".join(failures), file=sys.stderr)
        print("Install from requirements/train.txt and requirements/topology.txt.", file=sys.stderr)
        return 1
    if optional_missing:
        print(
            "\nMamba backend note: `mamba_ssm` is missing. "
            "This is expected on macOS/CPU. Use `model.mamba.backend=conv_fallback` for local smoke tests, "
            "or train the paper model on Linux with an NVIDIA GPU/CUDA/NVCC.",
            file=sys.stderr,
        )
    print("\nEnvironment verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
