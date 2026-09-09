from __future__ import annotations

import sys

import dibo


def test_package_version() -> None:
    assert dibo.__version__ == "0.1.0"


def test_import_does_not_load_gpu_packages() -> None:
    assert "vllm" not in sys.modules
    assert "torch" not in sys.modules
    assert "pynvml" not in sys.modules