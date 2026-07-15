import pytest

from korean_tts.device import resolve_device
from korean_tts.errors import UserFacingError


def test_resolve_cpu():
    assert resolve_device("cpu", cuda_available=lambda: True) == "cpu"


def test_resolve_cuda_when_available():
    assert resolve_device("cuda", cuda_available=lambda: True) == "cuda"


def test_resolve_cuda_when_unavailable_raises_clear_error():
    with pytest.raises(UserFacingError, match="CUDA was requested but is not available"):
        resolve_device("cuda", cuda_available=lambda: False)


def test_resolve_auto_prefers_cuda():
    assert resolve_device("auto", cuda_available=lambda: True) == "cuda"


def test_resolve_auto_falls_back_to_cpu():
    assert resolve_device("auto", cuda_available=lambda: False) == "cpu"


def test_resolve_rejects_unknown_device():
    with pytest.raises(UserFacingError, match="Unsupported device"):
        resolve_device("mps", cuda_available=lambda: False)
