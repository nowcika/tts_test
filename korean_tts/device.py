from __future__ import annotations

from collections.abc import Callable

from .errors import UserFacingError


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve_device(
    requested: str,
    cuda_available: Callable[[], bool] | None = None,
) -> str:
    check_cuda = cuda_available or _torch_cuda_available
    normalized = requested.strip().lower()

    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        if not check_cuda():
            raise UserFacingError(
                "CUDA was requested but is not available. Use --device cpu or install a CUDA-enabled PyTorch build."
            )
        return "cuda"
    if normalized == "auto":
        return "cuda" if check_cuda() else "cpu"

    raise UserFacingError("Unsupported device. Use one of: auto, cpu, cuda.")
