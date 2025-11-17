"""Device utilities for GPU/TPU support."""
from __future__ import annotations

import torch
from typing import Optional

# Try to import TPU support
try:
    import torch_xla.core.xla_model as xm
    import torch_xla
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False
    xm = None
    torch_xla = None


class DeviceConfig:
    """Configuration for device selection (GPU/TPU/CPU)."""

    def __init__(self, device_type: str = "auto"):
        """Initialize device configuration.

        Args:
            device_type: One of "auto", "gpu", "tpu", "cpu"
                - "auto": Prefer TPU > GPU > CPU
                - "gpu": Force GPU (CUDA)
                - "tpu": Force TPU (requires torch_xla)
                - "cpu": Force CPU
        """
        self.device_type = device_type.lower()
        self._device = None
        self._is_tpu = False

        if self.device_type == "tpu":
            if not TPU_AVAILABLE:
                raise RuntimeError(
                    "TPU requested but torch_xla not available. "
                    "Install with: pip install torch_xla"
                )
            self._device = xm.xla_device()
            self._is_tpu = True
        elif self.device_type == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("GPU requested but CUDA not available")
            self._device = torch.device("cuda")
            self._is_tpu = False
        elif self.device_type == "cpu":
            self._device = torch.device("cpu")
            self._is_tpu = False
        elif self.device_type == "auto":
            # Auto-detect: prefer TPU > GPU > CPU
            if TPU_AVAILABLE:
                try:
                    self._device = xm.xla_device()
                    self._is_tpu = True
                except Exception:
                    # TPU library available but no TPU runtime
                    if torch.cuda.is_available():
                        self._device = torch.device("cuda")
                        self._is_tpu = False
                    else:
                        self._device = torch.device("cpu")
                        self._is_tpu = False
            elif torch.cuda.is_available():
                self._device = torch.device("cuda")
                self._is_tpu = False
            else:
                self._device = torch.device("cpu")
                self._is_tpu = False
        else:
            raise ValueError(
                f"Unknown device_type: {device_type}. "
                f"Must be one of: auto, gpu, tpu, cpu"
            )

    @property
    def device(self) -> torch.device:
        """Get the PyTorch device object."""
        return self._device

    @property
    def is_tpu(self) -> bool:
        """Check if using TPU."""
        return self._is_tpu

    @property
    def is_gpu(self) -> bool:
        """Check if using GPU."""
        return not self._is_tpu and self._device.type == "cuda"

    @property
    def is_cpu(self) -> bool:
        """Check if using CPU."""
        return self._device.type == "cpu"

    def __str__(self) -> str:
        """String representation."""
        if self.is_tpu:
            return f"TPU (XLA): {self._device}"
        elif self.is_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            return f"GPU (CUDA): {gpu_name}"
        else:
            return "CPU"

    def mark_step(self):
        """Mark step for TPU (no-op on GPU/CPU).

        Call this after optimizer.step() when using TPU to ensure
        XLA graph compilation and execution.

        Uses torch_xla.sync(wait=True) instead of deprecated xm.mark_step().
        """
        if self._is_tpu and torch_xla is not None:
            torch_xla.sync(wait=True)

    def save(self, model_state: dict, path: str):
        """Save model state (handles TPU serialization).

        Args:
            model_state: State dict or checkpoint dict to save
            path: Path to save to
        """
        if self._is_tpu and xm is not None:
            # TPU: Use XLA-specific save
            xm.save(model_state, path)
        else:
            # GPU/CPU: Use standard PyTorch save
            torch.save(model_state, path)

    def all_reduce(self, tensor: torch.Tensor, reduce_type: str = "sum") -> torch.Tensor:
        """All-reduce tensor across devices (for distributed training).

        Args:
            tensor: Tensor to reduce
            reduce_type: One of "sum", "mean"

        Returns:
            Reduced tensor
        """
        if self._is_tpu and xm is not None:
            # TPU: Use XLA all-reduce
            if reduce_type == "sum":
                return xm.all_reduce(xm.REDUCE_SUM, tensor)
            elif reduce_type == "mean":
                reduced = xm.all_reduce(xm.REDUCE_SUM, tensor)
                return reduced / xm.xrt_world_size()
            else:
                raise ValueError(f"Unknown reduce_type: {reduce_type}")
        else:
            # GPU/CPU: No-op for single device (return as-is)
            return tensor


def get_device(device_type: str = "auto") -> DeviceConfig:
    """Get device configuration.

    Args:
        device_type: One of "auto", "gpu", "tpu", "cpu"

    Returns:
        DeviceConfig object

    Example:
        >>> device_cfg = get_device("auto")
        >>> print(device_cfg)  # "GPU (CUDA): Tesla V100" or "TPU (XLA): xla:0"
        >>> model = model.to(device_cfg.device)
    """
    return DeviceConfig(device_type)
