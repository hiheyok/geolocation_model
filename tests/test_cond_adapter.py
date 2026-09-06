"""Conditioning rides beside the retrieval vector without entering the cosine.

`runs/PYR_LEVELS.md` measured what happens when extra detail is put *into* the
retrieval vector: against a fixed `L0L1` bank, an extra level at weight 0.10
replaces 8.2% of the top-16 and flips the top-1 for 12.7% of queries while the
hit rate does not move at all (59.6% -> 59.6%). It reorders the ranking without
being about location. So the only place left for extra pixels is a path that is
not a cosine -- the model's own conditioning input, which is compared to
nothing.

Two properties have to hold, and both are the kind that fail silently.

**Identity at initialisation.** The goal is to fine-tune from trained weights,
so a checkpoint loaded into the conditioned model must behave *exactly* as it
did before the adapter existed -- not approximately. A zero gate gives that,
and `test_the_adapter_is_exactly_identity_at_init` asserts bit equality rather
than a tolerance.

**The gate must still train.** A zero gate times a zero table never trains and
looks exactly like an honest null, which this project has already been caught
by. Here the table is randomly initialised, so the gate has gradient from the
first step and the projection starts once the gate has moved. Both halves of
that are asserted.

The prefix check is the other silent failure: the k-NN cache addresses the
retrieval file's embedding space, and a joined cache whose first columns came
from a *different* build of that file leaves every neighbour the right row of
the wrong geometry, with every shape and every index still valid.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config                              # noqa: E402
import tile_math as tm                     # noqa: E402
from dataset import _check_retrieval_prefix  # noqa: E402
from encoders import StreetProj            # noqa: E402
from model import GeoAgent                 # noqa: E402

D_R, D_C, D = 768, 1536, 512


# --- the adapter --------------------------------------------------------------

def test_the_adapter_is_exactly_identity_at_init():
    """Bit equality, not a tolerance: a fine-tune must start *at* the trained
    point, not near it."""
    torch.manual_seed(0)
    plain = StreetProj(D_R, D)
    torch.manual_seed(0)
    cond = StreetProj(D_R, D, d_cond=D_C)
    x, c = torch.randn(4, D_R), torch.randn(4, D_C)
    with torch.no_grad():
        a, b = plain(x), cond(torch.cat([x, c], -1))
    assert torch.equal(a, b)


def test_the_conditioning_is_ignored_at_init_whatever_it_contains():
    torch.manual_seed(0)
    cond = StreetProj(D_R, D, d_cond=D_C)
    x = torch.randn(4, D_R)
    with torch.no_grad():
        a = cond(torch.cat([x, torch.zeros(4, D_C)], -1))
        b = cond(torch.cat([x, torch.randn(4, D_C) * 100], -1))
    assert torch.equal(a, b)


def test_the_gate_has_gradient_at_init():
    """Not the zero-gate deadlock: the table is random, so the gate moves."""
    torch.manual_seed(0)
    cond = StreetProj(D_R, D, d_cond=D_C)
    out = cond(torch.cat([torch.randn(4, D_R), torch.randn(4, D_C)], -1))
    out.sum().backward()
    assert cond.cond_gate.grad.abs().item() > 0
    # and the projection waits its turn -- zero while the gate is zero
    assert cond.cond_proj.weight.grad.abs().max() == 0


def test_the_projection_trains_once_the_gate_has_moved():
    torch.manual_seed(0)
    cond = StreetProj(D_R, D, d_cond=D_C)
    with torch.no_grad():
        cond.cond_gate.fill_(0.1)
    out = cond(torch.cat([torch.randn(4, D_R), torch.randn(4, D_C)], -1))
    out.sum().backward()
    assert cond.cond_proj.weight.grad.abs().max() > 0


def test_the_conditioning_changes_the_output_once_gated():
    torch.manual_seed(0)
    cond = StreetProj(D_R, D, d_cond=D_C)
    with torch.no_grad():
        cond.cond_gate.fill_(1.0)
    x = torch.randn(4, D_R)
    with torch.no_grad():
        a = cond(torch.cat([x, torch.zeros(4, D_C)], -1))
        b = cond(torch.cat([x, torch.randn(4, D_C)], -1))
    assert not torch.allclose(a, b)


def test_an_unconditioned_cache_is_refused():
    """Silently splitting a 768-d vector would take retrieval as conditioning."""
    cond = StreetProj(D_R, D, d_cond=D_C)
    with pytest.raises(ValueError, match="retrieval"):
        cond(torch.randn(4, D_R))


# --- the checkpoint path ------------------------------------------------------

def agent(**kw):
    return GeoAgent(d_street=D_R, n_actions=tm.actions(),
                    n_steps=tm.STEPS + 1, **kw)


def test_an_old_checkpoint_loads_and_only_the_adapter_is_new():
    torch.manual_seed(0)
    old = agent()
    torch.manual_seed(0)
    new = agent(d_cond=D_C)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert set(missing) == {
        "street.cond_gate", "street.cond_proj.weight", "street.cond_proj.bias",
        "street.cond_norm.weight", "street.cond_norm.bias"}
    assert not unexpected


def test_the_production_init_path_accepts_the_adapter():
    """REVIEW5 #6, and the reason the previous test was not enough.

    `--init` does not use `load_state_dict` directly: it filters the missing
    keys against an allowlist and exits on anything outside it. The adapter's
    five keys were not on that list, so the experiment's own intended command
    -- fine-tune the incumbent on a conditioned cache -- exited as an
    architecture mismatch while the suite stayed green, because the test above
    called PyTorch's loader and never came through the filter.

    This calls exactly what `train.main` calls.
    """
    from train import init_from
    torch.manual_seed(0)
    src = agent().state_dict()
    torch.manual_seed(0)
    init_from(agent(d_cond=D_C), src, "incumbent")     # must not raise


def test_the_production_init_path_still_rejects_a_real_mismatch():
    """The allowlist must not have become a blanket pass."""
    from train import init_from
    torch.manual_seed(0)
    src = agent().state_dict()
    del src["street.proj.weight"]                      # not an additive key
    with pytest.raises(SystemExit, match="does not match this architecture"):
        init_from(agent(d_cond=D_C), src, "broken")


def test_the_production_init_path_rejects_unexpected_parameters():
    from train import init_from
    torch.manual_seed(0)
    src = agent().state_dict()
    src["street.nonsense"] = torch.zeros(3)
    with pytest.raises(SystemExit, match="does not match this architecture"):
        init_from(agent(d_cond=D_C), src, "extra")


def test_the_retrieval_prior_never_sees_the_conditioning_block():
    """The learned keys live in the bank's space, which has no conditioning."""
    m = agent(d_cond=D_C, retr=True, retr_mode="dual")
    assert m.retr.__class__ is not None
    # the key projection was built for the retrieval width
    w = [p for n, p in m.retr.named_parameters() if n.endswith("weight")]
    assert any(p.shape[-1] == D_R for p in w), \
        "no retrieval-side projection is {}-d".format(D_R)


# --- the prefix guard ---------------------------------------------------------

def joined(tmp_path, monkeypatch, retr, cond, retr_name="r.f16.npy"):
    monkeypatch.setattr(config, "STREET_CACHE", tmp_path)
    # np.save only appends .npy when the name lacks it, and these names end
    # in .f16.npy already, so the file lands exactly where the loader looks.
    np.save(tmp_path / retr_name, retr.astype(np.float16))
    return np.concatenate([retr, cond], 1).astype(np.float16)


def test_a_matching_prefix_passes(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    r = rng.standard_normal((50, 8)).astype(np.float16)
    c = rng.standard_normal((50, 4)).astype(np.float16)
    s = joined(tmp_path, monkeypatch, r, c)
    _check_retrieval_prefix(tmp_path / "j.f16.npy", s, "r.f16.npy", 8)


def test_a_rebuilt_retrieval_cache_under_the_same_name_is_caught(
        tmp_path, monkeypatch):
    """The failure a string comparison cannot see.

    Same filename, same shape, same row count -- different vectors. The k-NN
    addresses one embedding space and the model's retrieval block is in
    another, and every neighbour index is still perfectly in range.
    """
    rng = np.random.default_rng(0)
    r = rng.standard_normal((50, 8)).astype(np.float16)
    c = rng.standard_normal((50, 4)).astype(np.float16)
    s = joined(tmp_path, monkeypatch, r, c)
    # the retrieval cache is rebuilt in place, keeping its name
    other = rng.standard_normal((50, 8)).astype(np.float16)
    np.save(tmp_path / "r.f16.npy", other)
    with pytest.raises(SystemExit, match="different space under the same name"):
        _check_retrieval_prefix(tmp_path / "j.f16.npy", s, "r.f16.npy", 8)


def test_a_missing_retrieval_cache_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STREET_CACHE", tmp_path)
    s = np.zeros((10, 12), np.float16)
    with pytest.raises(SystemExit, match="not on"):
        _check_retrieval_prefix(tmp_path / "j.f16.npy", s, "gone.f16.npy", 8)


def test_a_wrong_shaped_retrieval_cache_is_refused(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    r = rng.standard_normal((50, 8)).astype(np.float16)
    s = joined(tmp_path, monkeypatch, r,
               rng.standard_normal((50, 4)).astype(np.float16))
    with pytest.raises(SystemExit, match="declares"):
        _check_retrieval_prefix(tmp_path / "j.f16.npy", s, "r.f16.npy", 6)


def test_an_undeclared_retrieval_file_warns_rather_than_failing(
        tmp_path, monkeypatch, capsys):
    """Allowed, but never silent: without it nothing ties this file's
    retrieval half to the bank the neighbours came from."""
    monkeypatch.setattr(config, "STREET_CACHE", tmp_path)
    _check_retrieval_prefix(tmp_path / "j.f16.npy", np.zeros((5, 12),
                                                             np.float16),
                            None, 8)
    assert "cannot be verified" in capsys.readouterr().out


# --- REVIEW6 #1: neighbours carry the conditioning suffix too -----------------

@pytest.mark.parametrize("mode", ["scalar", "cond", "pos", "dual"])
def test_conditioned_prior_slices_query_and_neighbours(mode):
    """The keyed modes read neighbour *embeddings*, and those are gathered
    from the same combined street file as the query.

    Slicing only the query left the key projections -- built for the retrieval
    width -- receiving the wider tensor, and `pos`/`dual` raised on the first
    batch. `dual` is the mode this project actually trains with.

    The assertion is not merely "it runs": the conditioned call must equal the
    call made with explicit retrieval-width inputs, which is what establishes
    that the conditioning was dropped rather than folded in somewhere.
    """
    dr, dc, B, K = 8, 4, 3, 5
    torch.manual_seed(0)
    m = GeoAgent(d_street=dr, d_cond=dc, n_actions=tm.actions(),
                 n_steps=tm.STEPS + 1, retr=True, retr_mode=mode).eval()
    g = torch.Generator().manual_seed(1)
    nx = torch.randint(0, 16, (B, K), generator=g)
    ny = torch.randint(0, 16, (B, K), generator=g)
    sim = torch.rand(B, K, generator=g)
    emb_r = torch.randn(B, K, dr, generator=g)
    emb_c = torch.randn(B, K, dc, generator=g)
    st_r = torch.randn(B, dr, generator=g)
    st_c = torch.randn(B, dc, generator=g)
    x0 = torch.rand(B, generator=g)
    y0 = torch.rand(B, generator=g)
    step = torch.zeros(B, dtype=torch.long)
    n_logits = tm.actions()

    with torch.no_grad():
        wide = m.retr_prior((nx, ny, sim, torch.cat([emb_r, emb_c], -1)),
                            torch.cat([st_r, st_c], -1),
                            x0, y0, step, 1, n_logits)
        m.street.d_cond = 0                    # same model, prefixes supplied
        narrow = m.retr_prior((nx, ny, sim, emb_r), st_r,
                              x0, y0, step, 1, n_logits)
        m.street.d_cond = dc
    assert wide.shape[-1] == n_logits
    assert torch.equal(wide, narrow)


def test_the_unconditioned_prior_is_untouched():
    """The slice must be conditional -- a plain model has no suffix to drop."""
    dr, B, K = 8, 3, 5
    torch.manual_seed(0)
    m = GeoAgent(d_street=dr, n_actions=tm.actions(), n_steps=tm.STEPS + 1,
                 retr=True, retr_mode="dual").eval()
    g = torch.Generator().manual_seed(1)
    nbrs = (torch.randint(0, 16, (B, K), generator=g),
            torch.randint(0, 16, (B, K), generator=g),
            torch.rand(B, K, generator=g),
            torch.randn(B, K, dr, generator=g))
    with torch.no_grad():
        out = m.retr_prior(nbrs, torch.randn(B, dr, generator=g),
                           torch.rand(B, generator=g),
                           torch.rand(B, generator=g),
                           torch.zeros(B, dtype=torch.long), 1, tm.actions())
    assert out.shape == (B, tm.actions())


# --- REVIEW6 #2: a fixed sample skips the same rows forever -------------------

def test_a_change_outside_the_sampled_rows_is_caught_by_the_digest(
        tmp_path, monkeypatch):
    """The sample is deterministic, so what it misses it misses every run.

    Row 0 is chosen because it is outside the 64-row probe on a 1,000-row
    cache. Without a content digest this passed; with one it cannot.
    """
    import safeio
    rng = np.random.default_rng(0)
    r = rng.standard_normal((1000, 8)).astype(np.float16)
    c = rng.standard_normal((1000, 4)).astype(np.float16)
    s = joined(tmp_path, monkeypatch, r, c)
    digest = safeio.content_digest(tmp_path / "r.f16.npy")

    probe = np.unique(np.random.default_rng(0).integers(0, 1000, 64))
    assert 0 not in probe, "pick a row the sample does not reach"
    r2 = r.copy()
    r2[0] = r2[0] + 1
    np.save(tmp_path / "r.f16.npy", r2)

    # the sampled check alone still passes -- that is the bug
    _check_retrieval_prefix(tmp_path / "j.f16.npy", s, "r.f16.npy", 8)
    # the digest does not
    with pytest.raises(SystemExit, match="rebuilt under the same name"):
        _check_retrieval_prefix(tmp_path / "j.f16.npy", s, "r.f16.npy", 8,
                                want_digest=digest)


def test_a_matching_digest_passes(tmp_path, monkeypatch):
    import safeio
    rng = np.random.default_rng(0)
    r = rng.standard_normal((200, 8)).astype(np.float16)
    c = rng.standard_normal((200, 4)).astype(np.float16)
    s = joined(tmp_path, monkeypatch, r, c)
    _check_retrieval_prefix(tmp_path / "j.f16.npy", s, "r.f16.npy", 8,
                            want_digest=safeio.content_digest(
                                tmp_path / "r.f16.npy"))


def test_no_digest_warns_that_only_a_sample_was_checked(
        tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(0)
    r = rng.standard_normal((200, 8)).astype(np.float16)
    s = joined(tmp_path, monkeypatch, r,
               rng.standard_normal((200, 4)).astype(np.float16))
    _check_retrieval_prefix(tmp_path / "j.f16.npy", s, "r.f16.npy", 8)
    assert "only a 64-row sample" in capsys.readouterr().out


# --- NeighborBatch: the tuple that caused REVIEW6 #1 -------------------------

def test_neighbor_batch_accepts_legacy_sequences():
    """Three call sites built this tuple independently; all must still work."""
    from model import NeighborBatch
    x, y, sim = torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 3)
    emb = torch.zeros(2, 3, 8)
    assert NeighborBatch.of((x, y, sim)).emb is None
    assert NeighborBatch.of([x, y, sim, emb]).emb is emb
    nb = NeighborBatch(x, y, sim, emb)
    assert NeighborBatch.of(nb) is nb
    assert NeighborBatch.of(None) is None


def test_retrieval_only_slices_and_is_idempotent():
    """The operation that had to be remembered twice, now named once."""
    from model import NeighborBatch
    nb = NeighborBatch(torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 3),
                       torch.arange(2 * 3 * 12).float().reshape(2, 3, 12))
    cut = nb.retrieval_only(8)
    assert cut.emb.shape == (2, 3, 8)
    assert torch.equal(cut.emb, nb.emb[..., :8])
    assert cut.retrieval_only(8).emb.shape == (2, 3, 8)   # idempotent
    assert cut.x is nb.x and cut.sim is nb.sim            # nothing else moved


def test_retrieval_only_tolerates_no_embeddings():
    """`scalar` and `cond` modes pass no embeddings at all."""
    from model import NeighborBatch
    nb = NeighborBatch(torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 3))
    assert nb.retrieval_only(8).emb is None


# --- REVIEW6 #3: a rebuilt join does not launder a stale k-NN ----------------

class Cache:
    """An .npz stand-in: `files` plus item access, all `check_bytes` reads."""

    def __init__(self, **kw):
        self._d = kw
        self.files = list(kw)

    def __getitem__(self, k):
        return self._d[k]


def test_a_rebuilt_join_is_consistent_and_the_knn_is_still_stale(tmp_path,
                                                                 monkeypatch):
    """The step that makes REVIEW6 #3 invisible to every other check.

    Rebuild the retrieval cache under its own name, then rebuild the joined
    cache from it. The prefix check now *passes* -- it is comparing the new
    join against the new retrieval file, and they agree byte for byte. The
    k-NN built before the rebuild still describes neighbours chosen in the old
    vectors, and the only thing that can say so is the stamp it carries.
    """
    import knnmeta
    import safeio
    rng = np.random.default_rng(0)
    r_old = rng.standard_normal((200, 8)).astype(np.float16)
    c = rng.standard_normal((200, 4)).astype(np.float16)
    joined(tmp_path, monkeypatch, r_old, c)
    knn = Cache(street_digest=np.array(
                   safeio.content_digest(tmp_path / "r.f16.npy")))

    r_new = rng.standard_normal((200, 8)).astype(np.float16)
    s_new = joined(tmp_path, monkeypatch, r_new, c)

    # both halves of the join agree -- the prefix guard has nothing to say
    _check_retrieval_prefix(tmp_path / "j.f16.npy", s_new, "r.f16.npy", 8,
                            want_digest=safeio.content_digest(
                                tmp_path / "r.f16.npy"))
    # the cache is still addressing the space that is gone
    with pytest.raises(SystemExit, match="rebuilt under the same name"):
        knnmeta.check_bytes(knn, tmp_path / "r.f16.npy", "knn.npz")


def test_size_and_mtime_cannot_carry_this_contract(tmp_path, monkeypatch):
    """Why the stamp is a content digest and not the cheap stamp.

    `file_stamp` is the right identity for cache invalidation, where the
    failure guarded is an accidental rebuild. Here the artifact is an
    authority, and a same-shaped rebuild preserves size exactly while mtime is
    restorable -- as `os.utime`, a copy tool, or an archive restore will do.
    """
    import os

    import knnmeta
    import safeio
    rng = np.random.default_rng(0)
    p = tmp_path / "r.f16.npy"
    np.save(p, rng.standard_normal((200, 8)).astype(np.float16))
    stamp, digest = safeio.file_stamp(p), safeio.content_digest(p)

    st = p.stat()
    np.save(p, rng.standard_normal((200, 8)).astype(np.float16))
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))

    assert p.stat().st_size == st.st_size
    assert safeio.file_stamp(p) == stamp, "the cheap stamp sees nothing"
    assert safeio.content_digest(p) != digest
    with pytest.raises(SystemExit, match="rebuilt under the same name"):
        knnmeta.check_bytes(Cache(street_digest=np.array(digest)),
                            p, "k.npz")
