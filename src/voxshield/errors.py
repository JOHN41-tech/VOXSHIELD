"""Typed error hierarchy for VoxShield.

Every failure that can be caused by untrusted input is represented here as a
distinct type. The API layer maps these to stable ``status`` codes so clients
can branch on outcomes without parsing prose, and so that malformed input never
produces an unhandled 500.

The security property this module exists to protect: **a caller must never be
able to make the service crash, hang, or return a confident fraud verdict by
feeding it a hostile file.** Untrusted input ends in one of these exceptions.
"""

from __future__ import annotations


class VoxShieldError(Exception):
    """Base class for every error VoxShield raises deliberately."""


# ---------------------------------------------------------------------------
# Audio intake / decoding  (untrusted-input boundary)
# ---------------------------------------------------------------------------


class AudioIntakeError(VoxShieldError):
    """Base class for failures while accepting or decoding caller-supplied audio."""


class AudioTooLargeError(AudioIntakeError):
    """Upload exceeded the configured byte, duration, or channel budget.

    Raised *before* any large allocation happens, so an attacker cannot exhaust
    memory by declaring an enormous frame count in a WAV header.
    """


class UnsupportedAudioFormatError(AudioIntakeError):
    """The container or codec is not on the MVP allow-list.

    The MVP is WAV-only by deliberate decision (see ``docs/architecture.md``).
    Rejecting unknown containers by allow-list, rather than sniffing for known
    bad ones, means a novel container fails closed.
    """


class AudioDecodeError(AudioIntakeError):
    """The bytes claimed to be audio but could not be decoded.

    Covers malformed headers, truncated files, and codec-level failures inside
    libsndfile.
    """


class InvalidAudioSignalError(AudioIntakeError):
    """Decoded successfully but the signal is unusable.

    Raised for all-NaN, all-Inf, all-zero, or otherwise degenerate signals that
    would otherwise propagate NaN through the model and produce a meaningless
    score.
    """


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


class InsufficientSpeechError(VoxShieldError):
    """Too little usable speech to render any judgement.

    This is an **abstention**, not a low-risk verdict and definitely not a
    high-risk one. The pipeline reports it as ``INSUFFICIENT_SPEECH`` so the
    policy engine can ask for more audio rather than forcing a conclusion.

    The same code covers three distinct causes, and the message distinguishes
    them. Collapsing them into one wording produced messages that contradicted
    their own numbers -- "2.98s of speech, below the 2.00s minimum" -- which
    sends an operator to fix the wrong thing.
    """

    #: Not enough speech in total to justify a verdict.
    REASON_TOTAL = "total"
    #: Enough speech overall, but never in one contiguous stretch.
    REASON_CONTIGUOUS = "contiguous"
    #: Enough contiguous speech, but the whole clip is shorter than one window
    #: and the configured ``short_segment_policy`` is ``drop``.
    REASON_SHORT_WINDOW = "short_window"

    def __init__(
        self,
        speech_seconds: float,
        minimum_seconds: float,
        *,
        reason: str = REASON_TOTAL,
        longest_run_seconds: float | None = None,
        window_seconds: float | None = None,
    ) -> None:
        self.speech_seconds = speech_seconds
        self.minimum_seconds = minimum_seconds
        self.reason = reason
        self.longest_run_seconds = longest_run_seconds
        self.window_seconds = window_seconds

        if reason == self.REASON_CONTIGUOUS:
            detail = (
                f"Detected {speech_seconds:.2f}s of speech, but the longest "
                f"contiguous stretch is {longest_run_seconds:.2f}s, below the "
                f"{minimum_seconds:.2f}s minimum for a single window."
            )
        elif reason == self.REASON_SHORT_WINDOW:
            detail = (
                f"Detected {longest_run_seconds:.2f}s of contiguous speech, but "
                f"the recording is shorter than the {window_seconds:.2f}s analysis "
                f"window and the short-segment policy is 'drop'. Analyse at least "
                f"{window_seconds:.2f}s of audio, or use a policy of 'pad' or "
                f"'keep' to score a short recording."
            )
        else:
            detail = (
                f"Detected {speech_seconds:.2f}s of speech, "
                f"below the {minimum_seconds:.2f}s minimum required for a verdict."
            )
        super().__init__(detail)


class AudioQualityError(VoxShieldError):
    """Audio is decodable and non-degenerate, but too poor to score reliably."""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ModelUnavailableError(VoxShieldError):
    """No detector is loaded.

    Raised instead of returning a fabricated score. The MVP ships the
    preprocessing contract before any model is trained; an honest ``UNSCORED``
    response is strictly better than a plausible-looking guess, because a
    guessed score could trigger a step-up verification against a real customer.
    """


class ModelContractError(VoxShieldError):
    """A detector violated the expected output contract."""


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------


class PrivacyPolicyError(VoxShieldError):
    """A request violated a data-minimization or consent requirement.

    Example: a caller attempting to submit a direct identifier such as a phone
    number or account number, which the MVP explicitly does not accept.
    """
