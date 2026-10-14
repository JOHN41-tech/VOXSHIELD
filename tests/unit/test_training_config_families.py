"""Unit tests for the training configuration's model-family surface.

The Phase 4 additions -- a fourth family, two family sets, and eight new
:class:`ModelSpec` fields -- are configuration changes, and the failure mode of a
configuration change is quiet: a field that is accepted and ignored, a family that
loads but cannot be built, a set that omits the member its own name promises.

These tests are therefore about the surfaces rather than about any model. They
assert that the sets are consistent with each other and with the factory, that the
new fields round-trip, that unknown keys are still rejected, and that the defaults
are backwards-compatible with a configuration or artefact written before the
fields existed.
"""

from __future__ import annotations

import pytest

from voxshield.training.config import (
    MATRIX_FAMILIES,
    MODEL_FAMILIES,
    TORCH_FAMILIES,
    ModelSpec,
    TrainingError,
    parse_training_config,
)
from voxshield.training.models import build_baseline

CRNN_FIELDS = {
    "se_reduction": 8,
    "rnn_hidden": 32,
    "spec_augment": True,
    "freq_mask": 8,
    "time_mask": 16,
    "num_masks": 2,
    "label_smoothing": 0.05,
    "grad_clip": 1.0,
}


class TestFamilySets:
    def test_logmel_crnn_is_a_known_family(self) -> None:
        assert "logmel_crnn" in MODEL_FAMILIES

    def test_torch_families_are_all_known_families(self) -> None:
        # A family set naming something the validator rejects is a set whose
        # members cannot be constructed, which is worse than an empty one.
        assert TORCH_FAMILIES <= MODEL_FAMILIES
        assert MATRIX_FAMILIES <= MODEL_FAMILIES

    def test_matrix_families_are_all_torch_families(self) -> None:
        # Every matrix family today is torch-backed. If that stops being true,
        # this test is the place the two sets' relationship is reconsidered.
        assert MATRIX_FAMILIES <= TORCH_FAMILIES

    @pytest.mark.parametrize("family", sorted(TORCH_FAMILIES))
    def test_torch_families_report_uses_torch(self, family: str) -> None:
        assert ModelSpec(family=family).uses_torch is True

    @pytest.mark.parametrize("family", sorted(MATRIX_FAMILIES))
    def test_matrix_families_report_uses_matrices(self, family: str) -> None:
        assert ModelSpec(family=family).uses_matrices is True

    @pytest.mark.parametrize("family", ["mfcc_logreg", "mfcc_xgboost"])
    def test_tabular_families_report_neither(self, family: str) -> None:
        spec = ModelSpec(family=family)
        assert spec.uses_torch is False
        assert spec.uses_matrices is False

    @pytest.mark.parametrize("family", sorted(TORCH_FAMILIES))
    def test_every_torch_family_can_be_built(self, family: str) -> None:
        # The factory is the thing that has to agree with the set: a family in
        # TORCH_FAMILIES that build_baseline cannot construct would be an artefact
        # writer that accepts what the loader cannot rebuild.
        model = build_baseline(ModelSpec(family=family), n_frames=24, n_bands=12)
        assert model.requires_matrices is True


class TestCrnnFields:
    def test_defaults_are_conservative(self) -> None:
        spec = ModelSpec(family="logmel_crnn")
        assert spec.se_reduction == 8
        assert spec.rnn_hidden == 32
        assert spec.spec_augment is False
        assert spec.freq_mask == 8
        assert spec.time_mask == 16
        assert spec.num_masks == 2
        assert spec.label_smoothing == 0.0
        assert spec.grad_clip == 0.0

    def test_round_trips_through_to_dict(self) -> None:
        spec = ModelSpec(family="logmel_crnn", **CRNN_FIELDS)
        rebuilt = ModelSpec.from_mapping(spec.to_dict())
        assert rebuilt == spec

    def test_architecture_params_is_the_shape_affecting_subset(self) -> None:
        spec = ModelSpec(
            family="logmel_crnn",
            hidden_channels=(16, 32),
            dropout=0.3,
            se_reduction=4,
            rnn_hidden=16,
        )
        assert spec.architecture_params() == {
            "hidden_channels": [16, 32],
            "dropout": 0.3,
            "se_reduction": 4,
            "rnn_hidden": 16,
        }

    def test_architecture_params_excludes_training_knobs(self) -> None:
        # The recorded architecture must not grow a field whose value changes
        # nothing about the rebuild: an artefact that records the learning rate
        # invites a reader to think editing it changes the model.
        assert "spec_augment" not in ModelSpec(family="logmel_crnn").architecture_params()

    def test_unknown_model_option_is_still_rejected(self) -> None:
        with pytest.raises(TrainingError, match=r"unknown model option"):
            ModelSpec.from_mapping({"family": "logmel_crnn", "rnn_hiden": 16})

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("se_reduction", -1),
            ("rnn_hidden", -1),
            ("freq_mask", -1),
            ("time_mask", -1),
            ("num_masks", -1),
        ],
    )
    def test_negative_counts_are_rejected(self, key: str, value: int) -> None:
        with pytest.raises(TrainingError, match=key):
            ModelSpec(family="logmel_crnn", **{key: value})

    @pytest.mark.parametrize("value", [-0.1, 1.0, 1.5])
    def test_label_smoothing_outside_the_unit_interval_is_rejected(self, value: float) -> None:
        with pytest.raises(TrainingError, match="label_smoothing"):
            ModelSpec(family="logmel_crnn", label_smoothing=value)

    def test_negative_grad_clip_is_rejected(self) -> None:
        with pytest.raises(TrainingError, match="grad_clip"):
            ModelSpec(family="logmel_crnn", grad_clip=-1.0)

    def test_zero_disables_the_optional_blocks_without_error(self) -> None:
        spec = ModelSpec(family="logmel_crnn", se_reduction=0, rnn_hidden=0)
        assert spec.se_reduction == 0
        assert spec.rnn_hidden == 0


class TestBackwardsCompatibility:
    def test_a_configuration_without_crnn_keys_still_loads(self) -> None:
        # A Phase 3 YAML has no CRNN fields. Loading it must fill the defaults
        # rather than fail on the absence, or every existing recipe breaks.
        config = parse_training_config(
            {
                "model_id": "logmel_cnn",
                "model": {"family": "logmel_cnn", "hidden_channels": [32, 64]},
                "train": {"epochs": 2},
            }
        )
        assert config.model.family == "logmel_cnn"
        assert config.model.se_reduction == 8
        assert config.model.rnn_hidden == 32
        assert config.model.spec_augment is False

    def test_crnn_knobs_are_validated_for_every_family(self) -> None:
        # The check is not gated on the family, so a tabular recipe that sets a
        # depth for a block it does not have is caught rather than silently kept.
        with pytest.raises(TrainingError, match="se_reduction"):
            parse_training_config(
                {
                    "model_id": "mfcc_logreg",
                    "model": {"family": "mfcc_logreg", "se_reduction": -2},
                }
            )
