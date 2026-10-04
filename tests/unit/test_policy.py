"""Unit tests for the advisory action ladder and policy engine.

The single most important assertion in this file: with no model loaded, the
engine recommends *nothing*. A caller must be able to distinguish "we don't
know" from "we think it's fine", because collapsing those two is how an
unvalidated system ends up quietly approving fraud.
"""

from __future__ import annotations

import numpy as np
import pytest

from voxshield.errors import ModelUnavailableError
from voxshield.models.interface import SegmentScore, UnavailableDetector
from voxshield.policy.actions import (
    MIN_SPEECH_SECONDS,
    ActionType,
    RiskBand,
    actions_for,
)
from voxshield.policy.evaluator import (
    HIGH_THRESHOLD,
    MEDIUM_THRESHOLD,
    PolicyEngine,
    band_for_score,
)


class TestBandThresholds:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (0.0, RiskBand.LOW),
            (MEDIUM_THRESHOLD - 0.01, RiskBand.LOW),
            (MEDIUM_THRESHOLD, RiskBand.MEDIUM),
            (0.79, RiskBand.MEDIUM),
            (HIGH_THRESHOLD, RiskBand.HIGH),
            (1.0, RiskBand.HIGH),
        ],
    )
    def test_banding(self, score: float, expected: RiskBand) -> None:
        assert band_for_score(score) is expected

    @pytest.mark.parametrize("score", [-0.01, 1.01, float("nan"), float("inf")])
    def test_rejects_out_of_range(self, score: float) -> None:
        # A detector returning an out-of-range score is broken. Clamping would
        # hide the defect behind a plausible number.
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            band_for_score(score)


class TestActionLadder:
    def test_no_model_means_no_action(self) -> None:
        action = actions_for(band=RiskBand.HIGH, speech_seconds=5.0, model_version=None)
        assert action.action is ActionType.NONE
        assert action.band is RiskBand.UNKNOWN

    def test_low_band_only_logs(self) -> None:
        action = actions_for(band=RiskBand.LOW, speech_seconds=5.0, model_version="v1")
        assert action.action is ActionType.LOG_ONLY
        assert action.requires_human_review is False

    def test_medium_band_challenges_mfa(self) -> None:
        action = actions_for(band=RiskBand.MEDIUM, speech_seconds=5.0, model_version="v1")
        assert action.action is ActionType.CHALLENGE_MFA

    def test_high_band_escalates_to_human(self) -> None:
        action = actions_for(band=RiskBand.HIGH, speech_seconds=5.0, model_version="v1")
        assert action.action is ActionType.ESCALATE_TO_ANALYST
        assert action.requires_human_review is True

    def test_unknown_band_gives_no_action(self) -> None:
        action = actions_for(band=RiskBand.UNKNOWN, speech_seconds=5.0, model_version="v1")
        assert action.action is ActionType.NONE

    def test_never_blocks(self) -> None:
        # There is deliberately no blocking rung on the ladder.
        rungs = set(ActionType)
        assert "BLOCK" not in {r.value for r in rungs}
        assert "FREEZE" not in {r.value for r in rungs}
        assert ActionType.NONE is not None

    def test_too_little_speech_cannot_be_banded(self) -> None:
        # Programming error upstream: the pipeline should have abstained.
        with pytest.raises(ValueError, match="insufficient speech"):
            actions_for(
                band=RiskBand.HIGH,
                speech_seconds=MIN_SPEECH_SECONDS - 0.01,
                model_version="v1",
            )

    def test_rationale_disclaims_finding_about_caller(self) -> None:
        action = actions_for(band=RiskBand.HIGH, speech_seconds=5.0, model_version="v1")
        assert "not a finding about the caller" in action.rationale


class TestPolicyEngine:
    def _features(self, n: int = 2) -> list[np.ndarray]:
        return [np.zeros((100, 80), dtype=np.float32) for _ in range(n)]

    def test_unavailable_detector_yields_no_assessment(self) -> None:
        action, assessment = PolicyEngine().evaluate(
            detector=UnavailableDetector(),
            features=self._features(),
            speech_seconds=5.0,
        )
        assert assessment is None
        assert action.action is ActionType.NONE
        assert action.band is RiskBand.UNKNOWN

    def test_unavailable_detector_does_not_leak_details(self) -> None:
        action, _ = PolicyEngine().evaluate(
            detector=UnavailableDetector(),
            features=self._features(),
            speech_seconds=5.0,
        )
        # The rationale must not imply the audio was assessed.
        assert "no model" in action.rationale.lower()

    def test_high_score_escalates(self, detector) -> None:
        action, assessment = PolicyEngine().evaluate(
            detector=detector(score=0.95),
            features=self._features(),
            speech_seconds=5.0,
        )
        assert assessment is not None
        assert assessment.band is RiskBand.HIGH
        assert action.action is ActionType.ESCALATE_TO_ANALYST

    def test_low_score_only_logs(self, detector) -> None:
        action, assessment = PolicyEngine().evaluate(
            detector=detector(score=0.05),
            features=self._features(),
            speech_seconds=5.0,
        )
        assert assessment is not None
        assert assessment.band is RiskBand.LOW
        assert action.action is ActionType.LOG_ONLY

    def test_aggregation_uses_mean_not_max(self) -> None:
        # One spurious high window must not dominate a call-level verdict; that
        # is the failure mode that produces false accusations.
        class MixedDetector:
            model_version = "mixed-1.0.0"

            def score_segments(self, features):
                return [
                    SegmentScore(0, 0.05, self.model_version),
                    SegmentScore(1, 0.99, self.model_version),
                ]

        _, assessment = PolicyEngine().evaluate(
            detector=MixedDetector(),
            features=self._features(2),
            speech_seconds=5.0,
        )
        assert assessment is not None
        assert assessment.synthetic_probability == pytest.approx(0.52)

    def test_aggregation_averages_uniform_scores(self, detector) -> None:
        _, assessment = PolicyEngine().evaluate(
            detector=detector(score=0.7),
            features=self._features(3),
            speech_seconds=5.0,
        )
        assert assessment is not None
        assert assessment.synthetic_probability == pytest.approx(0.7)
        assert assessment.n_segments_scored == 3

    def test_broken_detector_yields_no_assessment(self, broken_detector) -> None:
        # An operational failure is not evidence of anything.
        action, assessment = PolicyEngine().evaluate(
            detector=broken_detector,
            features=self._features(),
            speech_seconds=5.0,
        )
        assert assessment is None
        assert action.action is ActionType.NONE

    def test_empty_segment_list_yields_no_assessment(self) -> None:
        class EmptyDetector:
            model_version = "empty-1.0.0"

            def score_segments(self, features):
                return []

        action, assessment = PolicyEngine().evaluate(
            detector=EmptyDetector(),
            features=[],
            speech_seconds=5.0,
        )
        assert assessment is None
        assert action.action is ActionType.NONE

    def test_bona_fide_is_complement(self, detector) -> None:
        _, assessment = PolicyEngine().evaluate(
            detector=detector(score=0.25),
            features=self._features(),
            speech_seconds=5.0,
        )
        assert assessment is not None
        assert assessment.bona_fide_probability == pytest.approx(0.75)

    def test_summary_carries_no_audio(self, detector) -> None:
        _, assessment = PolicyEngine().evaluate(
            detector=detector(score=0.9),
            features=self._features(3),
            speech_seconds=5.0,
        )
        assert assessment is not None
        summary = assessment.summary()
        for value in summary.values():
            assert not isinstance(value, (bytes, bytearray, np.ndarray))
        assert summary["n_segments_scored"] == 3

    def test_detector_payload_is_explicit_when_unavailable(self) -> None:
        payload = PolicyEngine.detector_payload(None)
        # Fields must be present-and-null, not omitted, so a caller can tell
        # "no score exists" from "field not sent".
        assert payload["status"] == "unavailable"
        assert payload["score"] is None
        assert payload["model_version"] is None
        assert payload["reason_code"] == "MODEL_UNAVAILABLE"

    def test_detector_payload_when_scored(self, detector) -> None:
        _, assessment = PolicyEngine().evaluate(
            detector=detector(score=0.9),
            features=self._features(),
            speech_seconds=5.0,
        )
        payload = PolicyEngine.detector_payload(assessment)
        assert payload["status"] == "ok"
        assert payload["score"] == pytest.approx(0.9)
        assert payload["reason_code"] == "SCORED"

    def test_policy_version_is_declared(self) -> None:
        assert PolicyEngine().version


class TestSegmentScoreContract:
    @pytest.mark.parametrize("score", [-0.1, 1.1])
    def test_rejects_out_of_range(self, score: float) -> None:
        from voxshield.errors import ModelContractError

        with pytest.raises(ModelContractError):
            SegmentScore(0, score, "v1")

    def test_unavailable_detector_raises_typed(self) -> None:
        with pytest.raises(ModelUnavailableError):
            UnavailableDetector().score_segments([])
