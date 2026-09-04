"""Weight decay was assigned by substring, and the substring lied.

`"pos" in name` reads as "positional embeddings, which should not decay".  What
it actually caught on the shipping arm was `retr.q_pos.weight` and
`retr.k_pos.weight` -- two `nn.Linear` projections, 196,608 parameters and 3.7%
of the model, which are the **learned retrieval keys**.  So the one branch that
`--retr-drop` exists to regularise was the one branch training with no weight
decay at all, and nothing in the run said so.

This is review item 29, and it changes what a trained arm is, which is why it
sat behind a decision rather than being fixed quietly.

The replacement classifies by what a tensor *is*.  These pin that it exempts
every table the old rule meant to exempt and nothing else -- the failure mode
of a fix like this is over-correcting into the sparse tables, where decoupled
decay would shrink each row in proportion to how rarely its parent tile is
sampled.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import tile_math as tm  # noqa: E402
from model import GeoAgent  # noqa: E402
from train import param_groups  # noqa: E402


def agent():
    """The shipping arm's shape, at a width that builds instantly."""
    return GeoAgent(d_street=64, n_actions=tm.actions(), n_steps=tm.STEPS + 1,
                    retr=True, retr_mode="dual", d_key=16, pool="attn")


def split(model):
    """(decayed names, exempt names)."""
    g = param_groups(model, 0.01)
    byid = {id(p): n for n, p in model.named_parameters()}
    return ({byid[id(p)] for p in g[0]["params"]},
            {byid[id(p)] for p in g[1]["params"]})


def test_the_learned_retrieval_keys_now_decay():
    """The whole point of the item. They are Linear projections, not tables."""
    dec, _ = split(agent())
    assert "retr.q_pos.weight" in dec
    assert "retr.k_pos.weight" in dec


def test_the_sparse_tables_are_still_exempt():
    """Decoupled decay runs every step regardless of gradient, so decaying a
    table biases it by how often each row's tile is sampled."""
    _, ex = split(agent())
    assert "map.pos.weight" in ex          # nn.Embedding
    assert "state.step.weight" in ex       # nn.Embedding


def test_norms_biases_and_scalars_are_still_exempt():
    m = agent()
    dec, ex = split(m)
    shapes = dict(m.named_parameters())
    assert all(shapes[n].ndim > 1 for n in dec), \
        "a bias or norm slipped into the decay group"
    assert "retr.w_pos" in ex               # a bare scalar


def test_every_trainable_parameter_lands_in_exactly_one_group():
    m = agent()
    dec, ex = split(m)
    live = {n for n, p in m.named_parameters() if p.requires_grad}
    assert dec | ex == live
    assert not (dec & ex)


def test_the_geo_table_is_exempt_but_its_query_projection_is_not():
    """The old rule carried a `"geo." in name` clause whose stated reason was
    the *sparsely refreshed table*.  It also caught `geo.q_geo`, a dense
    `nn.Linear` evaluated on every step, which the reason does not cover -- so
    the type test correctly narrows this clause rather than reproducing it.

    A second behaviour change, smaller than the retrieval keys: 8,192
    parameters, and only on the geo arms, which are not the shipping arm.
    """
    m = GeoAgent(d_street=64, n_actions=tm.actions(), n_steps=tm.STEPS + 1,
                 geo="key", d_geo=16)
    dec, ex = split(m)
    assert "geo.emb.weight" in ex          # nn.Embedding, sparsely refreshed
    assert "geo.gate" in ex                # a bare scalar
    assert "geo.q_geo.weight" in dec       # nn.Linear, dense, every step


def test_the_rule_reads_the_module_type_not_the_name():
    """A Linear called `pos` decays; an Embedding called anything does not."""
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_pos = nn.Linear(4, 4, bias=False)
            self.table = nn.Embedding(4, 4)
            self.norm = nn.LayerNorm(4)
            self.scale = nn.Parameter(torch.zeros(()))

    dec, ex = split(M())
    assert dec == {"q_pos.weight"}
    assert ex == {"table.weight", "norm.weight", "norm.bias", "scale"}


def test_the_legacy_rule_is_still_expressible_and_still_wrong():
    """Both sides of the comparison have to run fresh under one seed, because
    all 122 checkpoints on file were trained unseeded -- so scoring a new
    seeded arm against one of them would confound the seed with the
    treatment.  That needs the old behaviour to remain addressable."""
    from train import param_groups_legacy
    m = agent()
    g = param_groups_legacy(m, 0.01)
    byid = {id(p): n for n, p in m.named_parameters()}
    ex = {byid[id(p)] for p in g[1]["params"]}
    assert "retr.q_pos.weight" in ex and "retr.k_pos.weight" in ex


def test_the_two_rules_differ_only_where_the_item_says():
    m = agent()
    from train import param_groups_legacy
    byid = {id(p): n for n, p in m.named_parameters()}
    new_ex = {byid[id(p)] for p in param_groups(m, 0.01)[1]["params"]}
    old_ex = {byid[id(p)] for p in param_groups_legacy(m, 0.01)[1]["params"]}
    assert old_ex - new_ex == {"retr.q_pos.weight", "retr.k_pos.weight"}
    assert not new_ex - old_ex, "the fix must not exempt anything new"
