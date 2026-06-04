from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional

import torch
from torch import nn


def _safe_torch_load(path: str, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass
class _LayerAccumulator:
    count: int = 0
    absmax: Optional[torch.Tensor] = None
    min_val: Optional[torch.Tensor] = None
    max_val: Optional[torch.Tensor] = None
    sum_val: Optional[torch.Tensor] = None
    sumsq_val: Optional[torch.Tensor] = None

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        x = x.detach().to(torch.float32)
        c = int(x.shape[-1])
        x2d = x.reshape(-1, c)
        n = int(x2d.shape[0])

        cur_absmax = torch.amax(torch.abs(x2d), dim=0).cpu()
        cur_min = torch.amin(x2d, dim=0).cpu()
        cur_max = torch.amax(x2d, dim=0).cpu()
        cur_sum = torch.sum(x2d, dim=0).cpu()
        cur_sumsq = torch.sum(x2d * x2d, dim=0).cpu()

        if self.absmax is None:
            self.absmax = cur_absmax
            self.min_val = cur_min
            self.max_val = cur_max
            self.sum_val = cur_sum
            self.sumsq_val = cur_sumsq
        else:
            self.absmax = torch.maximum(self.absmax, cur_absmax)
            self.min_val = torch.minimum(self.min_val, cur_min)
            self.max_val = torch.maximum(self.max_val, cur_max)
            self.sum_val = self.sum_val + cur_sum  # type: ignore[operator]
            self.sumsq_val = self.sumsq_val + cur_sumsq  # type: ignore[operator]
        self.count += n

    def to_record(self) -> Dict[str, torch.Tensor | int]:
        if self.absmax is None or self.count <= 0:
            raise RuntimeError("empty calibration accumulator")
        mean = self.sum_val / max(self.count, 1)  # type: ignore[operator]
        var = self.sumsq_val / max(self.count, 1) - mean * mean  # type: ignore[operator]
        std = torch.sqrt(torch.clamp(var, min=0.0))
        shift = (self.max_val + self.min_val) * 0.5  # type: ignore[operator]
        return {
            "count": int(self.count),
            "absmax": self.absmax.contiguous(),
            "min": self.min_val.contiguous(),  # type: ignore[union-attr]
            "max": self.max_val.contiguous(),  # type: ignore[union-attr]
            "mean": mean.contiguous(),
            "std": std.contiguous(),
            "shift": shift.contiguous(),
        }


@dataclass
class DuQuantActivationCalibrator:
    """Collect per-Linear input activation statistics for OpenPI DuQuant PTQ."""

    include_regex: str = r".*"
    exclude_regex: str = ""
    accumulators: Dict[str, _LayerAccumulator] = field(default_factory=dict)
    _handles: list[Any] = field(default_factory=list)

    def _match(self, name: str) -> bool:
        if not re.search(self.include_regex, name):
            return False
        if self.exclude_regex and re.search(self.exclude_regex, name):
            return False
        return True

    def register(self, model: nn.Module) -> int:
        self.close()
        count = 0
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not self._match(name):
                continue
            self.accumulators.setdefault(name, _LayerAccumulator())

            def hook(_module: nn.Module, inputs: tuple[Any, ...], layer_name: str = name) -> None:
                if not inputs:
                    return
                x = inputs[0]
                if isinstance(x, torch.Tensor):
                    self.accumulators[layer_name].update(x)

            self._handles.append(module.register_forward_pre_hook(hook))
            count += 1
        return count

    def close(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "format": "openpi_duquant_activation_calib_v1",
            "layers": {name: acc.to_record() for name, acc in self.accumulators.items() if acc.count > 0},
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)


def collect_openpi_duquant_calibration(
    model: nn.Module,
    run_calibration: Callable[[nn.Module], Any],
    *,
    out_path: str,
    include_regex: str = r".*",
    exclude_regex: str = "",
) -> Dict[str, Any]:
    """Register hooks, run a user-provided OpenPI calibration rollout, save stats.

    `run_calibration(model)` must execute representative policy forwards using
    real LIBERO/OpenPI inputs.  This module intentionally does not invent a text
    dataset, because OpenPI action-head activations must be calibrated on policy
    inputs, not on generic language modeling samples.
    """
    calibrator = DuQuantActivationCalibrator(include_regex=include_regex, exclude_regex=exclude_regex)
    n = calibrator.register(model)
    print(f"[OPENPI-DUQUANT-CALIB] registered Linear hooks: {n}", flush=True)
    try:
        model.eval()
        with torch.inference_mode():
            run_calibration(model)
    finally:
        calibrator.close()

    state = calibrator.state_dict()
    calibrator.save(out_path)
    print(
        f"[OPENPI-DUQUANT-CALIB] saved layers={len(state['layers'])} path={out_path}",
        flush=True,
    )
    return state


def load_calibration(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"DuQuant calibration file not found: {path}")
    obj = _safe_torch_load(path, map_location="cpu")
    if isinstance(obj, dict) and "layers" in obj:
        return obj["layers"]
    if isinstance(obj, dict):
        # Accept direct {layer_name: stats} for debugging.
        return obj  # type: ignore[return-value]
    raise TypeError(f"Unsupported calibration file format: {type(obj)}")


def summarize_calibration(path: str, *, topk: int = 20) -> None:
    layers = load_calibration(path)
    rows = []
    for name, rec in layers.items():
        absmax = rec.get("absmax")
        if not isinstance(absmax, torch.Tensor):
            continue
        rows.append((float(absmax.max().item()), int(absmax.numel()), int(rec.get("count", 0)), name))
    rows.sort(reverse=True)
    print(f"[OPENPI-DUQUANT-CALIB] file={path} layers={len(rows)}")
    for max_abs, channels, count, name in rows[: int(topk)]:
        print(f"  max_abs={max_abs:12.5g} channels={channels:6d} samples={count:8d} {name}")
