"""Advisory action ladder.

VoxShield recommends; it does not enforce. Nothing in this module blocks a
transaction, freezes an account, or reports a caller to anyone. The strongest
action available is ``escalate_to_analyst``, which is a human being deciding
what happens next.

Two rules are enforced here rather than left to callers:

* **No signal, no action.** Without a model there is no evidence, so the ladder
  yields :data:`ActionType.NONE`. An unavailable detector must never produce a
  recommendation, because "we don't know" and "we think it's fine" are different
  answers and a caller that cannot tell them apart will treat the first as
  permission.
* **Insufficient speech is not a low risk result.** :func:`actions_for` refuses
  a band for an unusable clip instead of defaulting to ``LOW``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "NO_ACTION",
    "ActionType",
    "RecommendedAction",
    "RiskBand",
    "actions_for",
    "minimum_speech_seconds",
]


# The API is advisory throughout. A detection is a signal for a human or a
# downstream review step, never a verdict.
class ActionType(StrEnum):
    """What VoxShield recommends that the caller do next."""

    NONE = "none"
    LOG_ONLY = "log_only"
    CHALLENGE_MFA = "challenge_mfa"
    VERIFIED_CALLBACK = "verified_callback"
    ESCALATE_TO_ANALYST = "escalate_to_analyst"


class RiskBand(StrEnum):
    """Coarse band for a detection score.

    The band names a level of *suspicion*, not a conclusion about a person. A
    ``HIGH`` band means "a detector flagged this and a human should look", never
    "this caller committed fraud".
    """

    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# A clip shorter than this cannot support a detection, so no band may be
# assigned to it regardless of what a model returns.
MIN_SPEECH_SECONDS = 1.0

# Ordered weakest to strongest. Index in this tuple *is* the ladder position.
_LADDER: tuple[ActionType, ...] = (
    ActionType.LOG_ONLY,
    ActionType.CHALLENGE_MFA,
    ActionType.VERIFIED_CALLBACK,
    ActionType.ESCALATE_TO_ANALYST,
)

# Band -> ladder position. Deliberately not derived from the score here: the
# mapping is a business decision owned by the risk team, and Phase 0 ships it as
# an explicit table so that changing it is a reviewed code change rather than a
# threshold tweak that silently alters every decision.
_BAND_TO_LADDER_INDEX: Mapping[RiskBand, int] = {
    RiskBand.LOW: 0,
    RiskBand.MEDIUM: 1,
    RiskBand.HIGH: 3,
}


@dataclass(frozen=True, slots=True)
class RecommendedAction:
    """A single recommendation.

    Attributes:
        action: Recommended next step.
        band: Risk band that produced it, or ``UNKNOWN`` when unavailable.
        rationale: Human-readable justification. Must never contain content from
            the audio.
        requires_human_review: Whether a person must look before anything else
            happens. True for the strongest rung.
    """

    action: ActionType
    band: RiskBand
    rationale: str
    requires_human_review: bool


def minimum_speech_seconds() -> float:
    """Return the minimum speech duration required to band a clip."""
    return MIN_SPEECH_SECONDS


NO_ACTION = RecommendedAction(
    action=ActionType.NONE,
    band=RiskBand.UNKNOWN,
    rationale="No model is loaded, so no risk assessment was performed.",
    requires_human_review=False,
)


def _no_action(reason: str) -> RecommendedAction:
    return RecommendedAction(
        action=ActionType.NONE,
        band=RiskBand.UNKNOWN,
        rationale=reason,
        requires_human_review=False,
    )


def actions_for(
    *,
    band: RiskBand | None,
    speech_seconds: float,
    model_version: str | None,
    scores: Mapping[str, Any] | None = None,
) -> RecommendedAction:
    """Map a risk band to a recommended action.

    Args:
        band: Band from the detector, or ``None`` if detection did not run.
        speech_seconds: Duration of usable speech found by the VAD.
        model_version: Version of the detector, or ``None`` when unavailable.
        scores: Optional per-model scores, for the rationale string.

    Returns:
        A :class:`RecommendedAction`. Never a blocking action.

    Raises:
        ValueError: ``band`` is a valid :class:`RiskBand` but ``speech_seconds``
            claims usable speech below the minimum. This is a programming error
            upstream -- the pipeline should have produced
            ``INSUFFICIENT_SPEECH`` -- and is better caught than absorbed.
    """
    if not model_version:
        return _no_action(
            "No model is loaded. VoxShield reports audio quality only and recommends no action."
        )

    if band is None or band is RiskBand.UNKNOWN:
        return _no_action(
            "The detector produced no risk band for this clip. No action is recommended."
        )

    if speech_seconds < MIN_SPEECH_SECONDS:
        msg = (
            f"band={band.value!r} assigned with only {speech_seconds:.2f}s of "
            f"speech; the pipeline should have reported insufficient speech "
            f"instead of calling the model"
        )
        raise ValueError(msg)

    index = _BAND_TO_LADDER_INDEX.get(band)
    if index is None:
        return _no_action(f"No action is mapped to band {band.value!r}.")

    action = _LADDER[index]
    if band is RiskBand.LOW:
        rationale = (
            "Low suspicion. Recording the assessment is sufficient; no caller "
            "challenge is warranted."
        )
    elif band is RiskBand.MEDIUM:
        rationale = (
            "Moderate suspicion from an unverified detector. A step-up challenge "
            "is proportionate; this is not a finding about the caller."
        )
    else:
        rationale = (
            "High suspicion from an unverified detector. A human analyst should "
            "review before any further step. This is not a finding about the "
            "caller."
        )

    return RecommendedAction(
        action=action,
        band=band,
        rationale=rationale,
        requires_human_review=action is ActionType.ESCALATE_TO_ANALYST,
    )
