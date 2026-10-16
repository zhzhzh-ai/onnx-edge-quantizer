"""Deterministic per-tensor/per-channel uniform min-max quantization math."""

from dataclasses import asdict, dataclass
import numpy as np

DTYPES = {"int4": (4, True), "uint4": (4, False), "int8": (8, True),
          "uint8": (8, False), "int16": (16, True)}


@dataclass(frozen=True)
class QuantParams:
    dtype: str
    symmetric: bool
    scale: float
    zero_point: int
    qmin: int
    qmax: int
    observed_min: float
    observed_max: float

    def to_dict(self):
        return asdict(self)


def minmax_params(lo: float, hi: float, dtype: str, symmetric: bool) -> QuantParams:
    if dtype not in DTYPES:
        raise ValueError(f"Unknown dtype: {dtype}")
    if not np.isfinite([lo, hi]).all() or lo > hi:
        raise ValueError(f"Invalid finite min/max range: [{lo}, {hi}]")
    bits, signed = DTYPES[dtype]
    qmin, qmax = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    if symmetric:
        # Unsigned symmetric uses an offset midpoint, not zero_point=0.
        zp = 0 if signed else 1 << (bits - 1)
        radius = min(zp - qmin, qmax - zp)
        scale = max(abs(lo), abs(hi)) / radius
        qmin = zp - radius  # Narrow range: [-127,127] / [1,255], etc.
    else:
        low, high = min(lo, 0.0), max(hi, 0.0)
        scale = (high - low) / (qmax - qmin)
        zp = int(np.clip(np.rint(qmin - low / scale), qmin, qmax)) if scale else 0
    scale = float(np.float32(scale or 1.0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Quantization scale is outside the finite float32 range")
    return QuantParams(dtype, symmetric, scale, zp, qmin, qmax, float(lo), float(hi))


def quantize(values, params: QuantParams):
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Cannot quantize NaN or Inf")
    q = np.clip(np.rint(values / params.scale) + params.zero_point, params.qmin, params.qmax)
    dtype = np.int16 if params.dtype == "int16" else np.int8 if DTYPES[params.dtype][1] else np.uint8
    return q.astype(dtype)


def dequantize(values, params: QuantParams):
    return ((np.asarray(values).astype(np.float32) - params.zero_point) * params.scale).astype(np.float32)


def reconstruct_weight(values, dtype, symmetric, granularity="per-tensor", axis=0):
    """Quantize/reconstruct weights, collecting independent ranges per channel."""
    values = np.asarray(values, dtype=np.float32)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Weights must contain nonempty finite floating point values")
    if granularity == "per-tensor":
        params = minmax_params(float(values.min()), float(values.max()), dtype, symmetric)
        return dequantize(quantize(values, params), params), {
            **params.to_dict(), "granularity": granularity, "axis": None}
    if granularity != "per-channel":
        raise ValueError(f"Unsupported weight granularity: {granularity}")
    if not isinstance(axis, int) or not -values.ndim <= axis < values.ndim:
        raise ValueError(f"Invalid weight channel axis {axis} for shape {values.shape}")
    axis %= values.ndim
    channels = np.moveaxis(values, axis, 0)
    params = [minmax_params(float(c.min()), float(c.max()), dtype, symmetric) for c in channels]
    restored = np.stack([dequantize(quantize(c, p), p) for c, p in zip(channels, params)])
    return np.moveaxis(restored, 0, axis), {
        "dtype": dtype, "symmetric": symmetric, "granularity": granularity, "axis": axis,
        "scales": [p.scale for p in params], "zero_points": [p.zero_point for p in params],
        "channels": [p.to_dict() for p in params]}


def validate_native(weight_dtype, activation_dtype, weight_symmetric, activation_symmetric):
    problems = []
    if weight_dtype not in ("int4", "int8"):
        problems.append("native LiteRT weights must be int4 or int8")
    if activation_dtype not in ("int8", "int16"):
        problems.append("native internal activations must be int8 or int16 (uint8 I/O is a different setting)")
    if not weight_symmetric:
        problems.append("native integer kernels require symmetric weights")
    if activation_dtype == "int16" and not activation_symmetric:
        problems.append("native int16 activations require --activation-scheme symmetric")
    if problems:
        raise ValueError("; ".join(problems) + ". Use --mode reference for explicit float simulation of this combination.")
