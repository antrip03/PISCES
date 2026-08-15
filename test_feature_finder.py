"""
Unit tests for get_feature_activations's checkpoint/resume and names_filter
restriction, added after a real Kaggle T4 run hit a CUDA OOM inside this
function (run_with_cache_with_saes caching every hook point across the whole
model, not just the one this function reads) with no way to resume short of
redoing the much more expensive effect-measurement stage that precedes it.
No GPU or real SAE needed -- a minimal stub model provides just
run_with_cache_with_saes, matching the one method this function actually
calls on `model`.
"""
from __future__ import annotations

import sys
import types

import torch

# feature_finder.py imports editor.py (-> sae_lens) and evals.py (which pulls
# in transformers/datasets/openai/transformer_lens, all real, heavy, and in
# the case of sae_lens/transformer_lens observed to hang in this environment).
# Nothing under test here needs any of them -- stub both modules out.
if "sae_lens" not in sys.modules:
    fake_sae_lens = types.ModuleType("sae_lens")
    # get_feature_activations calls get_feature_saes -> the real (cached)
    # SAEConfig.get(), which calls SAE.from_pretrained(...)[0] -- stub it to
    # return a harmless placeholder; FakeModelForActivations below never
    # actually reads the `saes` argument it's passed, so the object's
    # content doesn't matter, only that construction doesn't crash.
    fake_sae_lens.SAE = type("SAE", (), {"from_pretrained": staticmethod(lambda release, sae_id, device: (object(), {}))})
    sys.modules["sae_lens"] = fake_sae_lens

if "evals" not in sys.modules:
    fake_evals = types.ModuleType("evals")
    for _name in (
        "eval_alpaca", "TransformerLensModel", "GeminiEvaluator", "evaluate_mmlu",
        "MCQAEvaluations", "evaluate_open_ended", "OpenEndedQuestion",
    ):
        setattr(fake_evals, _name, type(_name, (), {}))
    sys.modules["evals"] = fake_evals

from editor import Feature  # noqa: E402
from feature_finder import get_feature_activations  # noqa: E402


class FakeModelForActivations:
    """Stands in for model.run_with_cache_with_saes: returns a cache dict
    containing only the hook_sae_acts_post tensor(s) whose name passes
    names_filter -- mirrors real run_with_cache(_with_saes) behavior closely
    enough to test both the restriction and the checkpoint logic without a
    real model/SAE. `fire_batches`: dict[(layer, id)] -> set of batch
    CONTENTS (as tuples) at which that feature fires -- keyed by content, not
    call order/index, since a resumed run skips earlier batches entirely and
    a counter that just increments per-call would silently desync from the
    real forget_set position (an earlier version of this fake had exactly
    that bug -- always caught by test_checkpoint_resume, which is the whole
    reason to key on content instead of self-tracked position)."""

    def __init__(self, fire_batches: dict[tuple[int, int], set[tuple[str, ...]]]):
        self.fire_batches = fire_batches
        self.calls: list[tuple[list[str], object]] = []  # (batch, names_filter) per call
        # get_feature_saes (called internally by get_feature_activations)
        # reads model.cfg.tokenizer_name / model.cfg.device.
        self.cfg = types.SimpleNamespace(tokenizer_name="gemma-2-2b-it", device="cpu")

    def run_with_cache_with_saes(self, batch, saes, return_type=None, names_filter=None):
        batch_key = tuple(batch)
        self.calls.append((list(batch), names_filter))

        layers = {layer for layer, _ in self.fire_batches}
        cache = {}
        for layer in layers:
            hook_name = f"blocks.{layer}.hook_mlp_out.hook_sae_acts_post"
            if names_filter is not None and not names_filter(hook_name):
                continue
            max_id = max(fid for (l, fid) in self.fire_batches if l == layer)
            acts = torch.zeros(1, 1, max_id + 1)
            for (l, fid), fire_at in self.fire_batches.items():
                if l == layer and batch_key in fire_at:
                    acts[0, 0, fid] = 1.0
            cache[hook_name] = acts
        return None, cache


def test_names_filter_restricts_cache_to_only_needed_hooks():
    features = [Feature(layer=1, id=5, neg=False), Feature(layer=1, id=9, neg=True)]
    model = FakeModelForActivations({(1, 5): {("a", "b", "c")}, (1, 9): set()})

    get_feature_activations(model, features, ["a", "b", "c"], [" the"], [" and"], batch_size=3)

    assert len(model.calls) == 1
    _, names_filter = model.calls[0]
    assert names_filter("blocks.1.hook_mlp_out.hook_sae_acts_post") is True
    # a hook this function never reads must be excluded
    assert names_filter("blocks.5.hook_resid_post") is False
    assert names_filter("blocks.1.attn.hook_pattern") is False


def test_results_count_firings_correctly_across_batches():
    features = [Feature(layer=1, id=5, neg=False)]
    # 6 lines, batch_size=3 -> batches ["a","b","c"] and ["d","e","f"]; fires only in the first
    model = FakeModelForActivations({(1, 5): {("a", "b", "c")}})

    results = get_feature_activations(model, features, ["a", "b", "c", "d", "e", "f"], [" the"], [" and"], batch_size=3)

    assert results[(1, 5)] == 1
    assert len(model.calls) == 2


def test_checkpoint_resume_skips_completed_batches_and_accumulates(tmp_path):
    features = [Feature(layer=1, id=5, neg=False)]
    ckpt = str(tmp_path / "activations.ckpt")
    lines = ["a", "b", "c", "d", "e", "f"]  # batches ["a","b","c"] (start 0) and ["d","e","f"] (start 3)

    # Simulate "a prior run completed batch 0 (which fired once), checkpointed,
    # then crashed before batch 3" by hand-writing the checkpoint such a run
    # would have produced.
    from feature_finder import _save_checkpoint
    _save_checkpoint(ckpt, {"next_batch_start": 3, "results": {(1, 5): 1}})

    # Resume: the underlying feature fires again on the second batch this time
    # -- final count must be the OLD checkpointed 1 PLUS this run's new firing,
    # not just this run's own count, and the first batch must not be
    # re-processed (or even passed to the model) at all.
    model = FakeModelForActivations({(1, 5): {("d", "e", "f")}})
    results = get_feature_activations(model, features, lines, [" the"], [" and"], batch_size=3, checkpoint_path=ckpt)

    assert results[(1, 5)] == 2  # 1 (resumed) + 1 (this run's second-batch firing)
    assert len(model.calls) == 1  # only the un-completed batch was run
    assert model.calls[0][0] == ["d", "e", "f"]

    import os
    assert not os.path.exists(ckpt), "checkpoint should be cleaned up after successful completion"
