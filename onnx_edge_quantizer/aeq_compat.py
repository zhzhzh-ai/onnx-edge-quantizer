"""Scoped compatibility for AEQ per-tensor parameters shared across ranks."""
from contextlib import contextmanager
from dataclasses import replace
from threading import RLock

_rank_lock = RLock()


@contextmanager
def scalar_quantization_ranks():
    """Reshape singleton scale/zero arrays, retaining AEQ's normal validation.

    AEQ's rank fixer expands an already broadcast singleton array a second
    time when a Gather, Concat or PADV2 changes rank. A single encoding has
    exactly the same values at every rank. Channel/block encodings still use
    the original implementation. Restore the hook even if export fails.
    """
    from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as uq
    stats = {'scalar_parameter_rank_fixes': 0}
    with _rank_lock:
        original = uq.fix_quantization_params_rank

        def fix(data, params):
            if (params.block_size == 0 and params.scale.size == 1 and params.zero_point.size == 1
                    and (params.scale.ndim != data.ndim or params.zero_point.ndim != data.ndim)):
                stats['scalar_parameter_rank_fixes'] += 1
                shape = (1,) * data.ndim
                return replace(params, scale=params.scale.reshape(shape),
                               zero_point=params.zero_point.reshape(shape))
            return original(data, params)

        uq.fix_quantization_params_rank = fix
        try:
            yield stats
        finally:
            uq.fix_quantization_params_rank = original
