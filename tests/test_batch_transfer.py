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
