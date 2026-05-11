import math
import sys
import unittest
from pathlib import Path

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None
    nn = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bif.analysis.bif_runner import (  # noqa: E402
    _compute_nbeta_value,
    _resolve_observable_max_samples,
    _sampling_effective_batch_size,
)
from bif.config import SGLDConfig  # noqa: E402
from bif.training.sgld import LocalizedSGLDSampler  # noqa: E402


@unittest.skipIf(torch is None, "torch is not available in the current interpreter")
class SamplingAlignmentTest(unittest.TestCase):
    def test_sampling_effective_batch_size_matches_devinterp_inputs(self) -> None:
        cfg = SGLDConfig(
            nbeta=-1.0,
            nbeta_mode="devinterp",
            beta=0.25,
            gradient_accumulation_steps=4,
            batches_per_draw=7,
        )

        effective_batch_size = _sampling_effective_batch_size(
            train_batch_size=8,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        )
        self.assertEqual(effective_batch_size, 32)

        nbeta = _compute_nbeta_value(
            cfg,
            source_dataset_size=1000,
            sampling_effective_batch_size=effective_batch_size,
        )
        self.assertAlmostEqual(nbeta, 32 / math.log(32))

        dataset_mode_nbeta = _compute_nbeta_value(
            SGLDConfig(nbeta=-1.0, nbeta_mode="dataset", beta=0.125),
            source_dataset_size=800,
            sampling_effective_batch_size=effective_batch_size,
        )
        self.assertAlmostEqual(dataset_mode_nbeta, 100.0)

        self.assertEqual(_resolve_observable_max_samples(16, 3, explicit_limit=0), 48)
        self.assertEqual(_resolve_observable_max_samples(16, 3, explicit_limit=11), 11)

    def test_localized_sgld_reports_devinterp_style_diagnostics(self) -> None:
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)

        anchor = {"weight": torch.zeros_like(model.weight.data)}
        cfg = SGLDConfig(
            lr=1e-2,
            gamma=0.5,
            nbeta=2.0,
            weight_decay=0.25,
        )
        sampler = LocalizedSGLDSampler(
            model=model,
            anchor_params=anchor,
            config=cfg,
            source_dataset_size=32,
            effective_batch_size=8,
        )

        model.weight.grad = torch.ones_like(model.weight.data)
        diagnostics = sampler._sgld_update(torch.Generator().manual_seed(0))

        self.assertGreater(diagnostics.grad_norm, 0.0)
        self.assertGreater(diagnostics.scaled_grad_norm, 0.0)
        self.assertGreater(diagnostics.noise_norm, 0.0)
        self.assertGreater(diagnostics.localization_norm, 0.0)
        self.assertGreater(diagnostics.weight_decay_norm, 0.0)
        self.assertGreaterEqual(diagnostics.prior_norm, diagnostics.localization_norm)
        self.assertGreater(diagnostics.distance, 0.0)
        self.assertTrue(math.isfinite(diagnostics.dot_grad_prior))
        self.assertTrue(math.isfinite(diagnostics.dot_grad_noise))
        self.assertTrue(math.isfinite(diagnostics.dot_prior_noise))


if __name__ == "__main__":
    unittest.main()
