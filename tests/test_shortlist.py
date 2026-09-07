"""The two-stage search must return what an exhaustive one returns.

**The fixture is the interesting part.** This method works because the real
embedding is anisotropic -- 768 nominal dimensions carrying variance as if
there were ~48 -- so truncating to 128 keeps the directions that decide a
cosine. On isotropic Gaussian rows there is no such structure and truncation
throws away five sixths of the signal: an early version of this test used
`torch.randn` and reported 41/200 top-1 agreement, which would have condemned
a correct implementation. On 400,000 real bank rows the same code agrees
300/300.

So the bank here has a decaying spectrum. It is not the real distribution, but
it has the property the design depends on, and a fixture without it tests
something this code never claims to do.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from shortlist import Shortlist, exhaustive  # noqa: E402


def bank(n=4000, d=256, seed=0, decay=0.987):
    """Rows with a decaying spectrum, like a PCA-projected corpus.

    The decay is chosen so the first quarter of the columns hold ~83% of the
    variance, matching the real bank's 128-of-768 at 80.5%. The first version
    of this used 0.90, where the first quarter holds 100.0% -- truncation was
    then lossless, the renormalisation below was a no-op, and the test named
    after it could not fail. `scripts/mutate.py` is what said so.
    """
    g = torch.Generator().manual_seed(seed)
    scale = torch.pow(torch.tensor(decay), torch.arange(d, dtype=torch.float32))
    x = torch.randn(n, d, generator=g) * scale
    return (x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)).half()


def queries(m=60, d=256, seed=1, decay=0.987):
    return bank(m, d, seed, decay).float()


def test_it_returns_what_the_full_scan_returns():
    B = bank()
    sl = Shortlist(B, "cpu", dim=64, probe=128)
    agree = 0
    for q in queries():
        q = q[None] / q.norm()
        _, rows = sl.topk(q, 16)
        _, want = exhaustive(B, q.half(), 16, "cpu")
        agree += int(rows[0].item() == want[0].item())
    assert agree >= 59, "top-1 agreed on only {}/60".format(agree)


def test_the_scores_are_the_exact_ones_not_the_truncated_ones():
    """A caller uses these as an exhaustive search's, so they must BE those."""
    B = bank()
    sl = Shortlist(B, "cpu", dim=64, probe=256)
    q = queries(1)[0][None]
    q = q / q.norm()
    s, rows = sl.topk(q, 8)
    exact = (q @ B[rows.cpu()].float().T).flatten()
    assert torch.allclose(s, exact, atol=1e-5)


def test_a_deeper_probe_never_returns_worse_neighbours():
    """Monotonicity is the property that makes `probe` a safety dial."""
    B = bank()
    q = queries(1)[0][None]
    q = q / q.norm()
    best = None
    for probe in (16, 64, 256, 1024):
        s, _ = Shortlist(B, "cpu", dim=64, probe=probe).topk(q, 8)
        top = s[0].item()
        if best is not None:
            assert top >= best - 1e-6, "probe {} scored worse".format(probe)
        best = top


def test_probing_the_whole_bank_is_exact():
    B = bank(n=500)
    sl = Shortlist(B, "cpu", dim=32, probe=500)
    for q in queries(20):
        q = q[None] / q.norm()
        s, rows = sl.topk(q, 5)
        ws, want = exhaustive(B, q.half(), 5, "cpu")
        assert rows.tolist() == want.tolist()


def test_the_index_is_renormalised_after_truncation():
    """A truncated unit vector is not a unit vector; leaving it unnormalised
    ranks partly by how much magnitude survived the cut."""
    sl = Shortlist(bank(), "cpu", dim=64)
    norms = sl.index.float().norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=2e-3)
    raw = bank()[:, :64].float().norm(dim=1)
    assert raw.min() < 0.99, (
        "truncation does not shorten these rows, so the fixture cannot "
        "show that renormalising them matters")


def test_k_and_probe_are_clamped_to_the_bank():
    B = bank(n=12)
    sl = Shortlist(B, "cpu", dim=8, probe=999)
    q = queries(1)[0][None]
    s, rows = sl.topk(q / q.norm(), 50)
    assert len(rows) == 12 and len(s) == 12


def test_a_wider_request_than_the_bank_has_columns_is_clamped():
    sl = Shortlist(bank(d=64), "cpu", dim=512)
    assert sl.dim == 64


def test_isotropic_rows_are_where_this_does_not_work():
    """Recorded, not asserted as a goal: the method needs the structure the
    real embedding has, and this is the fixture that hides that."""
    g = torch.Generator().manual_seed(3)
    x = torch.randn(4000, 256, generator=g)
    B = (x / x.norm(dim=1, keepdim=True)).half()
    sl = Shortlist(B, "cpu", dim=64, probe=128)
    agree = 0
    for i in range(40):
        q = torch.randn(1, 256, generator=g)
        q = q / q.norm()
        _, rows = sl.topk(q, 8)
        _, want = exhaustive(B, q.half(), 8, "cpu")
        agree += int(rows[0].item() == want[0].item())
    assert agree < 40, (
        "isotropic rows now agree perfectly, so this fixture no longer "
        "demonstrates the dependence on anisotropy")


def test_only_a_projected_bank_may_be_truncated(tmp_path, monkeypatch):
    """The precondition was stated in the module docstring and enforced
    nowhere, so `--shortlist` defaulted on for every bank -- including the
    4,608-d raw concatenation, whose first 128 columns are the first crop's
    first 128 features, ordered by the encoder's layout rather than by
    variance. Review measured top-1 matching exhaustive search on 122 of 128
    queries there, with one true winner at rank 181 against a 128-candidate
    cutoff, so a deeper probe would not rescue it either.
    """
    import json

    import shortlist

    raw = tmp_path / "dual_c3.f16.npy"
    raw.write_bytes(b"")
    ok, why = shortlist.suitable(raw)
    assert ok is False and "no projection" in why

    proj = tmp_path / "pca768.f16.npy"
    proj.write_bytes(b"")
    (tmp_path / "pca768.f16.npy.prov.json").write_text(
        json.dumps({"projection": "pca768_pca.npz", "rows": 4}),
        encoding="utf-8")
    ok, why = shortlist.suitable(proj)
    assert ok is True and "pca768_pca.npz" in why


def test_the_check_reads_provenance_not_the_width():
    """A 768-d file is not necessarily projected and a 1536-d one is not
    necessarily raw; the sidecar is what knows."""
    import inspect

    import shortlist

    src = inspect.getsource(shortlist.suitable)
    assert "projection_of" in src
    assert "shape" not in src and "768" not in src
