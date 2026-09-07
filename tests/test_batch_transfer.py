"""Neighbour indices must not make a round trip to the GPU and back.

`run_epoch` moved every tensor in the batch to the training device, including
`nbr_row`. `gather_nbr` then moved those indices straight back, because it
gathers on the TABLE's device and the street table lives on the host whenever
it does not fit in VRAM. The result was CPU -> GPU -> CPU -> gather -> GPU for
data that never needed to leave the host, and the download forces a
synchronisation that blocks the next batch's transfer (REVIEW9 #1).

`nbr_row` has exactly three consumers and all three only pass it to
`gather_nbr`, so it never belongs on the device at all.

The test asserts placement and equality, not speed: a throughput claim needs a
benchmark, and this is the part that can be established statically.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import HOST_ONLY, gather_nbr, to_device  # noqa: E402


def batch():
    return {"street": torch.randn(4, 8), "nbr_row": torch.arange(8).view(4, 2),
            "step": torch.zeros(4, dtype=torch.long), "tag": "not a tensor"}


def test_host_only_entries_stay_on_the_host():
    b = to_device(batch(), "cpu")
    assert b["nbr_row"].device.type == "cpu"
    assert "nbr_row" in HOST_ONLY


def test_everything_else_is_moved_and_non_tensors_pass_through():
    b = to_device(batch(), "cpu")
    assert torch.is_tensor(b["street"]) and torch.is_tensor(b["step"])
    assert b["tag"] == "not a tensor"


def test_the_gather_is_unchanged_by_where_the_indices_start():
    """The whole point: removing the round trip must not change a single row."""
    table = torch.randn(8, 5)
    rows = torch.arange(8).view(4, 2)
    a = gather_nbr(table, rows, "cpu")
    b = gather_nbr(table, rows.clone(), "cpu")
    assert torch.equal(a, b)
    assert torch.equal(a, table[rows].float())


def test_gather_still_works_when_the_table_is_where_the_rows_are_not():
    """`gather_nbr` decides by the table's device, which is what lets the
    caller stop guessing."""
    table = torch.randn(8, 5)
    rows = torch.arange(8).view(4, 2)
    out = gather_nbr(table, rows, "cpu")
    assert out.shape == (4, 2, 5) and out.dtype == torch.float32


class _Rows:
    """A stand-in for the index tensor that records how it was uploaded.

    The property under test is a keyword argument on a CUDA copy, which no
    CPU-only run reaches. Recording the call is what makes it checkable here
    rather than only on a machine with a card.
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.calls = []

    def to(self, device, **kw):
        self.calls.append((device, kw))
        return _Rows(device)


class _Table:
    def __init__(self, device):
        self.device = device

    def __getitem__(self, idx):
        return _Moved()


class _Moved:
    def to(self, dev, **kw):
        return self

    def float(self):
        return self


def test_the_index_upload_to_a_gpu_table_stays_non_blocking():
    """`nbr_row` used to reach the card through the batch transfer, which sets
    `non_blocking`. Keeping the row on the host moved that upload into
    `gather_nbr`, and without the flag an async copy out of pinned memory
    becomes a synchronising one on every step of a GPU-table run."""
    from dataset import gather_nbr

    rows = _Rows("cpu")
    gather_nbr(_Table("cuda"), rows, "cuda")
    assert rows.calls, "the index was never uploaded"
    device, kw = rows.calls[0]
    assert device == "cuda"
    assert kw.get("non_blocking") is True, (
        "the index upload is blocking; it was asynchronous before this change")


def test_a_host_table_does_not_move_the_index_at_all():
    """The point of the change: no round trip when the table is on the host."""
    from dataset import gather_nbr

    rows = _Rows("cpu")
    gather_nbr(_Table("cpu"), rows, "cuda")
    assert rows.calls == []


def test_every_batch_transfer_goes_through_to_device():
    """`to_device`'s docstring names three call sites -- `train.run_epoch`,
    `cond_probe` and `diag_beam` -- as the reason it exists. `cond_probe` was
    not migrated, so it kept the exact round trip this change removes, and the
    docstring was describing an intention rather than the code."""
    import re

    pat = re.compile(r"\.to\(dev[^)]*\)\s*if\s+torch\.is_tensor")
    bad = []
    for path in (sorted((ROOT / "scripts").glob("*.py"))
                 + sorted((ROOT / "src").glob("*.py"))):
        if path.name == "dataset.py":
            continue                      # where to_device itself lives
        for m in pat.finditer(path.read_text(encoding="utf-8")):
            bad.append(path.name)
    assert not bad, (
        "moves a batch to the device by hand instead of calling to_device, so "
        "nbr_row makes the round trip: " + ", ".join(sorted(set(bad))))
