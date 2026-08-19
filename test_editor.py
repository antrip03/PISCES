"""
Unit tests for replace_mlp_rows's debug_log_noop_edits opt-in diagnostic
(added to investigate the "No changes made to the model in layer X"
assertion -- see track_a_feature_discovery/README.md for the investigation
writeup). No GPU, real model, or real SAE needed: replace_mlp_rows only
ever touches model.blocks[layer].mlp.W_out, so a minimal stand-in with a
real torch tensor there is sufficient.

Two things are checked: (1) debug_log_noop_edits=False (the default) is
byte-for-byte the original behavior -- still asserts on a no-op edit, still
applies real edits and reverts them on exit; (2) debug_log_noop_edits=True
correctly distinguishes an empty-switches no-op (e.g. a dead SAE encoder
column) from a non-empty-but-storage-collapsed no-op (e.g. an fp16
downcast rounding an fp32-confirmed edit away), instead of crashing either
way.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch

# editor.py does `from sae_lens import SAE` at module level purely for other
# functions' type hints / SAE loading -- replace_mlp_rows never touches SAE
# at all. sae_lens's own import chain is heavy (transformer_lens registry
# init) and has been observed to hang in this environment; stub it out so
# these tests don't depend on it being importable at all.
if "sae_lens" not in sys.modules:
    fake_sae_lens = types.ModuleType("sae_lens")
    fake_sae_lens.SAE = type("SAE", (), {})
    sys.modules["sae_lens"] = fake_sae_lens

import editor  # noqa: E402
from editor import Feature, SAEConfig, get_mlp_act_signs, replace_mlp_rows  # noqa: E402


def make_fake_model(w_out: torch.Tensor):
    block = types.SimpleNamespace(mlp=types.SimpleNamespace(W_out=w_out))
    return types.SimpleNamespace(blocks={0: block})


def test_default_behavior_unchanged_asserts_on_empty_switches():
    w_out = torch.randn(4, 3)
    model = make_fake_model(w_out.clone())

    with pytest.raises(AssertionError, match="No changes made"):
        with replace_mlp_rows(model, {0: []}):
            pass


def test_default_behavior_unchanged_applies_and_reverts_real_edit():
    w_out = torch.randn(4, 3)
    original = w_out.clone()
    model = make_fake_model(w_out.clone())
    new_row = original[1] + 5.0  # unambiguously not a no-op

    with replace_mlp_rows(model, {0: [(1, new_row)]}):
        assert torch.allclose(model.blocks[0].mlp.W_out[1], new_row)
        assert not torch.allclose(model.blocks[0].mlp.W_out, original)

    # reverted on exit
    assert torch.allclose(model.blocks[0].mlp.W_out, original)


def test_debug_mode_logs_and_continues_on_empty_switches(capsys):
    w_out = torch.randn(4, 3)
    model = make_fake_model(w_out.clone())

    with replace_mlp_rows(model, {0: []}, debug_log_noop_edits=True):
        pass  # must NOT raise

    out = capsys.readouterr().out
    assert "0 switches were computed at all" in out
    assert "dead SAE encoder column" in out


def test_debug_mode_logs_and_continues_on_storage_collapsed_edit(capsys):
    # A "new" row that's fp32-different from old by torch.allclose's default
    # tolerance, but rounds to an identical fp16 value once both are cast --
    # mirrors the get_hswaps_full_signed .to(model_dtype) downcast collapsing
    # an edit that was already confirmed non-negligible in fp32.
    old_row = torch.tensor([1.0, 2.0, 3.0])
    delta = old_row * 2e-4  # fails fp32 allclose (rtol=1e-5), collapses in fp16
    new_row = old_row + delta
    assert not torch.allclose(new_row, old_row), "test setup: delta should be fp32-visible"
    assert torch.allclose(new_row.half(), old_row.half()), "test setup: delta should collapse in fp16"

    w_out = old_row.unsqueeze(0).repeat(4, 1).clone()
    model = make_fake_model(w_out.to(torch.float16))

    with replace_mlp_rows(model, {0: [(0, new_row.to(torch.float16))]}, debug_log_noop_edits=True):
        pass  # must NOT raise

    out = capsys.readouterr().out
    assert "switch(es) were computed and assigned" in out
    assert "index=0" in out


def test_debug_mode_logs_feature_id_when_layer_features_provided(capsys):
    w_out = torch.randn(4, 3)
    model = make_fake_model(w_out.clone())
    dead_feature = Feature(layer=0, id=4242, neg=False)

    with replace_mlp_rows(model, {0: []}, debug_log_noop_edits=True, layer_features={0: [dead_feature]}):
        pass  # must NOT raise

    out = capsys.readouterr().out
    assert "id=4242" in out
    assert "unknown -- layer_features not provided" not in out


def test_debug_mode_without_layer_features_falls_back_to_unknown(capsys):
    # layer_features omitted entirely (steer_features always passes it, but
    # replace_mlp_rows can still be called directly) -- must not crash, and
    # should say plainly that feature identity isn't available rather than
    # silently omitting it.
    w_out = torch.randn(4, 3)
    model = make_fake_model(w_out.clone())

    with replace_mlp_rows(model, {0: []}, debug_log_noop_edits=True):
        pass

    out = capsys.readouterr().out
    assert "unknown -- layer_features not provided" in out


def test_saeconfig_get_caches_by_release_sae_id_device(monkeypatch):
    """get_hswaps_full(_signed) call SAEConfig(...).get() once per candidate
    feature PER BATCH (via unlearn_concept) -- for a real discovery run
    that's thousands of calls for the exact same SAE. Without caching, each
    one is a fresh SAE.from_pretrained load with fresh GPU tensors, which
    was observed causing a CUDA OOM a few batches into a real Kaggle run
    (thousands of alloc/free cycles fragmenting the caching allocator).
    This confirms the fix: same (release, sae_id, device) returns the exact
    same object, and only calls the real loader once; a different layer
    (a genuinely different SAE) is NOT conflated with it."""
    editor._SAE_CACHE.clear()
    calls = []

    class FakeLoadedSAE:
        def float(self):
            return self  # mirrors sae_lens SAEs already being fp32 (see editor.py's comment)

    def fake_from_pretrained(release, sae_id, device):
        calls.append((release, sae_id, device))
        return (FakeLoadedSAE(), {"dummy": "metadata"})

    monkeypatch.setattr(editor, "SAE", types.SimpleNamespace(from_pretrained=fake_from_pretrained))

    cfg = SAEConfig(model_name="gemma-2-2b-it", layer=1, type="mlp", size="16k", device="cpu")
    first = cfg.get()
    second = cfg.get()
    third = SAEConfig(model_name="gemma-2-2b-it", layer=1, type="mlp", size="16k", device="cpu").get()

    assert first is second is third
    assert len(calls) == 1, f"expected exactly one real load, got {len(calls)}: {calls}"

    other = SAEConfig(model_name="gemma-2-2b-it", layer=2, type="mlp", size="16k", device="cpu").get()
    assert other is not first
    assert len(calls) == 2


def test_debug_mode_does_not_suppress_real_edits():
    w_out = torch.randn(4, 3)
    original = w_out.clone()
    model = make_fake_model(w_out.clone())
    new_row = original[2] + 5.0

    with replace_mlp_rows(model, {0: [(2, new_row)]}, debug_log_noop_edits=True):
        assert torch.allclose(model.blocks[0].mlp.W_out[2], new_row)

    assert torch.allclose(model.blocks[0].mlp.W_out, original)


class FakeModelForActSigns:
    """Stands in for model.run_with_cache: returns a cache dict containing
    only the hook tensors whose name passes names_filter -- mirrors the real
    run_with_cache closely enough to test the restriction without a real
    model. Includes a hook get_mlp_act_signs never reads (an attention
    pattern) so the test can prove names_filter actually excludes it, not
    just that a filter callable was passed."""

    def __init__(self, n_layers=2, d_mlp=3):
        self.cfg = types.SimpleNamespace(n_layers=n_layers, d_mlp=d_mlp)
        self.calls: list[object] = []  # names_filter per call
        self._token_ids = {" a": 1, " b": 2}
        self._line_tokens = {"a": 1, "b": 2}

    def to_single_token(self, token):
        return self._token_ids[token]

    def to_tokens(self, batch):
        return torch.tensor([[self._line_tokens[line]] for line in batch])

    def run_with_cache(self, batch, return_type=None, names_filter=None):
        self.calls.append(names_filter)
        cache = {}
        for layer in range(self.cfg.n_layers):
            hook_name = f"blocks.{layer}.mlp.hook_post"
            if names_filter is None or names_filter(hook_name):
                cache[hook_name] = torch.zeros(len(batch), 1, self.cfg.d_mlp)
        untouched_hook = "blocks.0.attn.hook_pattern"
        if names_filter is None or names_filter(untouched_hook):
            cache[untouched_hook] = torch.zeros(1)
        return None, cache


def test_get_mlp_act_signs_names_filter_excludes_unread_hooks():
    """Real Kaggle T4 run hit a CUDA OOM inside get_mlp_act_signs
    (run_with_cache caching every hook point across the whole model --
    attention patterns, residual stream, etc. -- not just the
    blocks.{layer}.mlp.hook_post tensors this function reads), 19/83 batches
    in: '14.54 GiB memory in use' of '14.56 GiB' total -- the same ceiling
    get_feature_activations hit before its own names_filter fix. This
    verifies the fix actually restricts the cache instead of just changing a
    comment."""
    model = FakeModelForActSigns(n_layers=2, d_mlp=3)

    # Callers (discover.py/run_kaggle.py) always pass an already-split list
    # of lines (forget_text.splitlines()[:1000]), not a raw string.
    get_mlp_act_signs(model, [" a"], ["a", "b"], batch_size=1)

    assert len(model.calls) == 2, "one run_with_cache call per batch (2 lines, batch_size=1)"
    for names_filter in model.calls:
        assert names_filter is not None, "names_filter must be passed, not left as the caching-everything default"
        assert names_filter("blocks.0.mlp.hook_post")
        assert names_filter("blocks.1.mlp.hook_post")
        assert not names_filter("blocks.0.attn.hook_pattern"), "must exclude hooks this function never reads"
