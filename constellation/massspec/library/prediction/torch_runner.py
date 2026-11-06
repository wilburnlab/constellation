"""Running a TorchScript model over many inputs, the same way every time.

Inputs are cut into fixed batches in their given order (callers pass
requests in canonical order), each batch is run under
``torch.inference_mode`` on one intra-op thread, and ``threads`` batches
run at once on a thread pool (TorchScript releases the GIL, and a model
run concurrently on one intra-op thread per call does not share state).

That makes the output depend on the batch size, never on the number of
threads: a batch is the same arithmetic wherever it runs. (Measured on
JChronologer's models, torch 2.12: changing torch's intra-op thread count
moves values by up to 4.5e-6, and changing the batch size by up to 2.2e-5;
the same batch size gives the same bits.) The batch size is therefore a
setting a prediction records, and the thread count is not.

Reused by every in-process model (JChronologer's, and the in-house one to
come); nothing here knows a model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np
import torch

__all__ = ["DEFAULT_BATCH_SIZE", "run_batched", "torch_threads"]

#: JChronologer's default (``ChronologerOptions.DEFAULT_BATCH_SIZE`` is 2048).
DEFAULT_BATCH_SIZE = 2048


@contextmanager
def torch_threads(n: int) -> Iterator[None]:
    """torch's intra-op thread count set to ``n`` for a block, then restored."""
    before = torch.get_num_threads()
    torch.set_num_threads(max(1, int(n)))
    try:
        yield
    finally:
        torch.set_num_threads(before)


def run_batched(
    model: Callable[..., torch.Tensor],
    inputs: Sequence[np.ndarray],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    threads: int = 1,
    device: str = "cpu",
) -> np.ndarray:
    """``model(*batch)`` over ``inputs`` (arrays sharing a first axis), in
    batches of ``batch_size``, concatenated as one float array.

    The output of each batch is moved to the CPU and kept as the model gave
    it (float32 for JChronologer's).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    n = len(inputs[0])
    if any(len(x) != n for x in inputs):
        raise ValueError("inputs must share their first axis")
    starts = list(range(0, n, batch_size))
    if not starts:
        raise ValueError("no inputs to run (callers skip empty requests)")

    def one(start: int) -> np.ndarray:
        tensors = [torch.from_numpy(np.ascontiguousarray(x[start : start + batch_size])).to(device) for x in inputs]
        with torch.inference_mode():
            out = model(*tensors)
        return out.detach().cpu().numpy()

    with torch_threads(1):
        if threads <= 1 or len(starts) == 1:
            parts = [one(s) for s in starts]
        else:
            with ThreadPoolExecutor(max_workers=int(threads)) as pool:
                parts = list(pool.map(one, starts))
    return np.concatenate(parts, axis=0)
