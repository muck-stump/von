"""Compute device detection shared across Von.

These helpers used to live inside the superseded cross-encoder backend, which
made every consumer import a benchmark baseline just to resolve a device. They
are model-agnostic, so they live on their own now.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Union

import torch


class OpenVINODevice:
    """Represents an Intel compute device targeted via OpenVINO."""

    type = "openvino"
    index = None

    def __init__(self, target: str = "GPU", device_name: Optional[str] = None):
        self.target = target.upper()
        if device_name is None:
            device_name = _get_openvino_device_name(self.target)
        self.device_name = device_name or f"Intel {self.target}"

    def __repr__(self) -> str:
        return f"device(type='openvino', target='{self.target}')"

    def __str__(self) -> str:
        return f"openvino:{self.target.lower()}"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            o = other.lower().strip()
            return o in (
                "openvino",
                "ov",
                "intel",
                "intel_gpu",
                f"openvino:{self.target.lower()}",
                f"ov:{self.target.lower()}",
            )
        return isinstance(other, OpenVINODevice) and self.target == other.target


DeviceType = Union[torch.device, OpenVINODevice]


def is_openvino_available() -> bool:
    """Return True if OpenVINO Python runtime is installed."""
    try:
        import openvino as ov  # noqa: F401
        return True
    except ImportError:
        return False


def is_openvino_gpu_available() -> bool:
    """Return True if OpenVINO is installed and detects an Intel GPU."""
    try:
        import openvino as ov
        core = ov.Core()
        return "GPU" in core.available_devices
    except Exception:
        return False


def _get_openvino_device_name(target: str = "GPU") -> Optional[str]:
    """Fetch full device description string from OpenVINO runtime."""
    try:
        import openvino as ov
        core = ov.Core()
        if target in core.available_devices:
            return str(core.get_property(target, "FULL_DEVICE_NAME"))
    except Exception:
        pass
    return None


def _detect_device(device_str: Optional[str] = None) -> DeviceType:
    d_str = (device_str or os.environ.get("VON_DEVICE", "auto")).lower().strip()

    # Handle AMD ROCm / HIP aliases
    if d_str in ("rocm", "hip"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "AMD ROCm requested, but PyTorch CUDA/ROCm is not available. "
                "Ensure PyTorch was installed with ROCm support."
            )
        return torch.device("cuda")

    # Handle DirectML (Windows AMD / Intel)
    if d_str in ("dml", "directml"):
        try:
            import torch_directml  # type: ignore[import-not-found]  # optional extra
            return torch_directml.device()
        except ImportError:
            raise RuntimeError(
                "DirectML requested, but 'torch-directml' is not installed. "
                "Run 'pip install torch-directml'."
            )

    # Handle explicit OpenVINO device requests
    if d_str in ("openvino", "ov", "intel", "intel_gpu", "openvino:gpu", "ov:gpu"):
        if not is_openvino_available():
            raise RuntimeError(
                "OpenVINO requested, but 'openvino' is not installed. "
                "Run 'pip install openvino' or 'pip install von-sdk[intel]'."
            )
        return OpenVINODevice("GPU")

    if d_str in ("openvino:cpu", "ov:cpu"):
        if not is_openvino_available():
            raise RuntimeError(
                "OpenVINO requested, but 'openvino' is not installed. "
                "Run 'pip install openvino'."
            )
        return OpenVINODevice("CPU")

    if d_str != "auto":
        return torch.device(d_str)

    # Auto-detection priority:
    # 1. First-party discrete / primary accelerators (NVIDIA CUDA / AMD ROCm)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # 2. Apple Silicon Metal Performance Shaders
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    # 3. Intel GPU via OpenVINO (Iris Xe, Arc, Ultra iGPU/dGPU)
    if is_openvino_gpu_available():
        return OpenVINODevice("GPU")
    # 4. OpenVINO CPU runtime: ~1.6x the PyTorch CPU encoder on the same cores,
    #    now that independent_options checkpoints have their own traced graph.
    #    This is the path a CPU-only self-hosted endpoint should land on
    #    without anyone remembering to set VON_DEVICE.
    if is_openvino_available():
        return OpenVINODevice("CPU")
    # 5. PyTorch CPU fallback
    return torch.device("cpu")


def get_device_description(device: Any) -> str:
    if isinstance(device, OpenVINODevice) or (isinstance(device, str) and device.lower().startswith("openvino")):
        dev_name = getattr(device, "device_name", None) or _get_openvino_device_name("GPU") or "Intel GPU"
        return f"Intel GPU [OpenVINO: {dev_name}]"
    if hasattr(device, "type"):
        if device.type == "cuda":
            dev_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CUDA"
            if getattr(torch.version, "hip", None) or any(w in dev_name.lower() for w in ["amd", "radeon", "instinct"]):
                return f"AMD GPU [ROCm: {dev_name}]"
            return f"NVIDIA GPU [CUDA: {dev_name}]"
        elif device.type == "mps":
            return "Apple Silicon [MPS]"
        elif str(device).startswith("privateuseone"):
            return "DirectML GPU [AMD/Intel]"
    return "CPU"
