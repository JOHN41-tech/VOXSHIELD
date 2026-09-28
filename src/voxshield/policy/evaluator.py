"""Policy engine: turn detector output into an advisory recommendation.

Kept separate from the HTTP layer so the decision path can be tested without a
request, and separate from the detector so that "the model said 0.93" and "the
policy says escalate a human" are independently auditable.

Policy version is embedded in every response and every audit record. Without it
a stored decision cannot be replayed against the rules that produced it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from voxshield.audio.segmentation import aggregate_segment_scores
from voxshield.errors import ModelUnavailableError
from voxshield.models.interface import SegmentScore, SynthSpeechDetector
from voxshield.policy.actions import (
    RecommendedAction,
    RiskBand,
    actions_for,
)

__all__ = [
    "HIGH_THRESHOLD",
    "MEDIUM_THRESHOLD",
    "POLICY_VERSION",
    "ClipAssessment",
    "PolicyEngine",
    "band_for_score",
]

# Bump on any change to the action ladder or the band thresholds. Every
# evaluation result and audit record carries this string, so a stored decision is
# always attributable to the rules that produced it.
POLICY_VERSION = "mvp-1.0.0"

# Score at or above which a detection is MEDIUM, and at or above which it is
# HIGH. Phase 0 ships no model, so these are never exercised against a real
# detection. They are stated explicitly rather than defaulted, so that loading a
# model cannot leave the thresholds implicit.
MEDIUM_THRESHOLD = 0.50
HIGH_THRESHOLD = 0.80


def band_for_score(score: float) -> RiskBand:
    """Map a synthetic-speech score to a risk band.

    Args:
        score: Probability that the audio is synthetic, in ``[0, 1]``.

    Returns:
        The corresponding band.

    Raises:
        ValueError: Score outside ``[0, 1]``. A detector returning an
            out-of-range score is broken, and silently clamping it would hide
            that defect behind a plausible number.
    """
    if not 0.0 <= score <= 1.0:
        msg = f"score must be within [0, 1], got {score!r}"
        raise ValueError(msg)
    if score >= HIGH_THRESHOLD:
        return RiskBand.HIGH
    if score >= MEDIUM_THRESHOLD:
        return RiskBand.MEDIUM
    return RiskBand.LOW


@dataclass(frozen=True, slots=True)
class ClipAssessment:
    """Aggregate detector output for one clip.

    Attributes:
        model_version: Version of the weights that produced the scores.
        synthetic_probability: Aggregate score for the whole clip.
        n_segments_scored: Number of windows the detector returned.
        per_segment: Per-window scores, in window order.
        band: Band assigned from the aggregate score.
    """

    model_version: str
    synthetic_probability: float
    n_segments_scored: int
    per_segment: tuple[SegmentScore, ...]
    band: RiskBand

    @property
    def bona_fide_probability(self) -> float:
        """Complement of the synthetic probability.

        Assumes a two-class detector, which is what the CNN baseline in
        ``docs/architecture.md`` is specified to be. A detector with more classes
        must override this rather than let a misleading complement ship.
        """
        return 1.0 - self.synthetic_probability

    def summary(self) -> dict[str, Any]:
        """Compact, metadata-only summary for logs and audit records."""
        return {
            "model_version": self.model_version,
            "synthetic_probability": round(self.synthetic_probability, 6),
            "bona_fide_probability": round(self.bona_fide_probability, 6),
            "n_segments_scored": self.n_segments_scored,
            "segment_synthetic_probabilities": [
                round(s.synthetic_probability, 6) for s in self.per_segment
            ],
            "risk_band": self.band.value,
        }


def _aggregate(segment_scores: Sequence[SegmentScore]) -> float:
    """Aggregate per-window scores into one clip-level score.

    Delegates to
    :func:`voxshield.audio.segmentation.aggregate_segment_scores` so there is a
    single definition of how windows combine into a call-level score. That
    function uses the **mean**, not the max: a single spurious high window must
    not dominate a call-level verdict, because that is the failure mode that
    produces a confident false accusation against a real customer.

    The cost of the mean is a dilution effect -- one synthetic window in a long
    call can be pulled below the band threshold. The trade-off is deliberate and
    reversible: the response carries every per-window score, so a case where
    dilution occurred is visible and reviewable rather than hidden inside a
    single aggregate.
    """
    return aggregate_segment_scores(
        np.asarray([s.synthetic_probability for s in segment_scores], dtype=np.float64)
    )


class PolicyEngine:
    """Applies the advisory action ladder to a detector's output."""

    version = POLICY_VERSION

    def evaluate(
        self,
        *,
        detector: SynthSpeechDetector,
        features: list[Any],
        speech_seconds: float,
    ) -> tuple[RecommendedAction, ClipAssessment | None]:
        """Score a clip and recommend an action.

        A detector that is unavailable or fails at runtime yields no assessment
        and an action of ``none``, rather than a fabricated score.

        Args:
            detector: Detector to consult. :class:`UnavailableDetector` is the
                Phase 0 default and raises rather than scoring.
            features: Per-window log-mel arrays, in window order.
            speech_seconds: Duration of usable speech found by the VAD.

        Returns:
            The recommendation, and the assessment if scoring succeeded.
        """
        try:
            segment_scores = detector.score_segments(features)
        except ModelUnavailableError:
            action = actions_for(
                band=None,
                speech_seconds=speech_seconds,
                model_version=None,
            )
            return action, None
        except Exception:
            # A detector that raises is an operational failure, not evidence of
            # anything. Recommend nothing, and let the caller's own health
            # monitoring surface the failure via latency and error rate.
            action = actions_for(
                band=None,
                speech_seconds=speech_seconds,
                model_version=None,
            )
            return action, None

        if not segment_scores:
            action = actions_for(
                band=None,
                speech_seconds=speech_seconds,
                model_version=detector.model_version,
            )
            return action, None

        aggregate = _aggregate(segment_scores)
        band = band_for_score(aggregate)
        assessment = ClipAssessment(
            model_version=detector.model_version,
            synthetic_probability=aggregate,
            n_segments_scored=len(segment_scores),
            per_segment=tuple(segment_scores),
            band=band,
        )
        action = actions_for(
            band=band,
            speech_seconds=speech_seconds,
            model_version=detector.model_version,
            scores=assessment.summary(),
        )
        return action, assessment

    @staticmethod
    def detector_payload(assessment: ClipAssessment | None) -> dict[str, Any]:
        """Build the detector section of an API response.

        With no model, the probabilities and version are ``null`` and
        ``status`` is ``"unavailable"``. They are not omitted: a caller must be
        able to distinguish "no score exists" from "the field was not sent".
        """
        if assessment is None:
            return {
                "status": "unavailable",
                "model_version": None,
                "score": None,
                "bona_fide_probability": None,
                "synthetic_probability": None,
                "n_segments_scored": 0,
                "risk_band": RiskBand.UNKNOWN.value,
                "reason_code": "MODEL_UNAVAILABLE",
            }
        payload = assessment.summary()
        payload["status"] = "ok"
        payload["score"] = payload["synthetic_probability"]
        payload["reason_code"] = "SCORED"
        return payload

    @staticmethod
    def action_payload(action: RecommendedAction) -> dict[str, Any]:
        """Build the action section of an API response."""
        return {
            "action": action.action.value,
            "risk_band": action.band.value,
            "rationale": action.rationale,
            "requires_human_review": action.requires_human_review,
        }
