"""Coarse-to-fine bank search: a small resident index proposes, the exact
vectors decide.

A brute-force scan of the shipping bank is 3,400,180 x 768 fp16 = 5.22 GB.
That does not fit beside the model on an 8 GB card, so `serve.py` held it in
host RAM and shipped every block over PCIe **per query** -- measured at
~435 ms for one photograph, which is essentially the whole latency of the demo.

The embedding is far more compressible than its width suggests. On the
shipping bank the variance is spread as if it had ~48 dimensions:

    explained variance   128d 80.5%   256d 89.5%   384d 94.0%   512d 96.9%
    participation ratio  48 of 768

Because these are PCA coordinates the components are already ordered by
variance, so truncating to `dim` IS the optimal linear projection to `dim` --
nothing to fit, no basis to store.

Truncation does not lose the right neighbours; it REORDERS them near the cut,
which an exact rescore of a slightly longer shortlist repairs. Measured on
3,400,180 rows and 5,000 held-out queries, as the fraction of the exact 768-d
top-32 that the compressed index returns in its top-k:

    dim   recall@32   recall@64   recall@128   recall@256
    384      92.9%       99.9%       100.0%       100.0%
    256      87.9%       99.4%        99.9%       100.0%
    128      76.8%       94.6%        99.0%        99.7%

End to end -- shortlist, then exact rescore of the candidates -- the km
metrics are identical to exhaustive search at every threshold and the top-1
agrees on 99.96% of queries. That is why 128 is the default despite the worst
recall@32 in the table: what matters is recall at the PROBE depth, and the
rescore fixes the order of whatever survives.

The win is not the 6x smaller footprint. It is that 0.87 GB fits on the card
where 5.22 GB did not, which turns a ~12 GB/s PCIe scan into a ~450 GB/s VRAM
one.

**This changes latency, not accuracy.** Ranking, not coverage, is what limits
this system: 33.2% of queries at 25 km have a correct row the ranker does not
pick. An index returning 99.9% of the same rows cannot move that, and a result
that appears to should be disbelieved before it is published.
"""

import torch

DIM = 128
PROBE = 128


class Shortlist:
    """Two-stage top-k over an L2-normalised bank.

    `bank` is `(n, d)` fp16 and already normalised, on host or device. The
    index is built from it once; the bank itself stays where it is and is read
    only for the rows the shortlist proposes.
    """

    def __init__(self, bank, dev, dim=DIM, probe=PROBE, block=1 << 20):
        self.bank, self.dev = bank, dev
        self.dim = min(int(dim), bank.shape[1])
        self.probe = int(probe)
        self.index = torch.empty((bank.shape[0], self.dim),
                                 dtype=torch.float16, device=dev)
        for lo in range(0, bank.shape[0], block):
            hi = min(lo + block, bank.shape[0])
            # Truncate, then RE-NORMALISE. A truncated unit vector is not a
            # unit vector, and a cosine over unnormalised rows ranks partly by
            # how much of each row's magnitude happened to survive the cut.
            b = bank[lo:hi, :self.dim].to(dev, torch.float32)
            self.index[lo:hi] = (
                b / b.norm(dim=1, keepdim=True).clamp_min(1e-6)).half()

    @property
    def gb(self):
        return self.index.numel() * 2 / 1e9

    def topk(self, q, k):
        """`(scores, rows)` for one normalised query, exact scores.

        The returned similarities are the full-width ones, so a caller can use
        them as it used an exhaustive search's -- they are the same numbers
        for the rows both return.
        """
        n = self.index.shape[0]
        k = min(k, n)
        probe = min(max(self.probe, k), n)

        qd = q[:, :self.dim].to(self.dev, torch.float16)
        qd = qd / qd.float().norm(dim=1, keepdim=True).clamp_min(1e-6).half()
        cand = (qd @ self.index.T).float().flatten().topk(probe).indices

        # Exact rescore, at full width, of the shortlist only. The gather is
        # `probe` rows -- 197 KB at 128 candidates -- against a scan of the
        # whole bank, so the second stage is free next to the first.
        # The gather runs where the bank lives: indexing a host tensor with a
        # CUDA index raises, and moving a GPU bank's rows to the host to index
        # them would undo the point of holding it there.
        rows = cand if self.bank.device.type != "cpu" else cand.cpu()
        exact = self.bank[rows].to(self.dev, torch.float32)
        sims = (q.to(self.dev, torch.float32) @ exact.T).flatten()
        s, j = sims.topk(k)
        return s, cand[j]


def exhaustive(bank, q, k, dev, block=200_000):
    """Full scan, for the reference path and for tests to compare against."""
    n = bank.shape[0]
    out = torch.empty(n, dtype=torch.float32)
    qd = q.to(dev, torch.float16)
    for lo in range(0, n, block):
        blk = bank[lo:lo + block]
        blk = blk if blk.device.type == dev else blk.to(dev, non_blocking=True)
        out[lo:lo + block] = (qd @ blk.T).float().flatten().cpu()
    return out.topk(min(k, n))
