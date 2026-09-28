"""Unit tests for file validation and rejection reasons.

The properties tested here are the ones whose failure is invisible in a happy-path
build. A corpus that silently drops files still produces a manifest, a split, and
a headline number -- and nothing in those artefacts says the corpus is a third
smaller than the operator's disk. So the tests concentrate on three things:

* every refused file is accounted for, with a code from a closed vocabulary and a
  reason that names the measurement that failed;
* a file that fails several checks is reported under the most fundamental one, so
  an operator is sent to the threshold that was actually the constraint;
* ordering is deterministic, because a build whose rejection list reorders between
  runs cannot be diffed against the run before it.

Audio is synthesised rather than read from fixtures. A synthetic waveform has a
duration, a level, and a speech proportion the test chooses exactly, which is what
these tests are about; a fixture would add container-format questions and
encode/decode noise to properties that do not involve the samples.
"""

from __future__ import annotations

import math
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import soundfile as sf

from voxshield.audio.quality import (
    ISSUE_CLIPPED,
    ISSUE_SILENT,
    QualityReport,
)
from voxshield.config import AudioConfig
from voxshield.data.config import DataConfig, DatasetEntry, ValidationConfig
from voxshield.data.discovery import build_inventory, discover
from voxshield.data.errors import DatasetBuildError, DatasetConfigError
from voxshield.data.schema import UNKNOWN, SourceRecord
from voxshield.data.validation import (
    REJECT_AUDIO_BUDGET,
    REJECT_DUPLICATE_CONTENT,
    REJECT_DUPLICATE_FILE,
    REJECT_FILE_TOO_LARGE,
    REJECT_MISSING_METADATA,
    REJECT_NO_SPEECH,
    REJECT_QUALITY_WARNING,
    REJECT_TOO_LONG,
    REJECT_TOO_SHORT,
    REJECT_UNDECODABLE,
    REJECT_UNREADABLE,
    REJECT_UNSUPPORTED_FORMAT,
    REJECT_UNUSABLE_SIGNAL,
    REJECTION_CODES,
    Acceptance,
    FileMeasurement,
    Rejection,
    ValidationResult,
    measure_file,
    validate_measurement,
    validate_sources,
)

BONA_FIDE = "bona_fide"

SAMPLE_RATE = 16_000

# Validation reads exactly one Phase 1 field -- ``min_speech_seconds``, the floor a
# file must clear to fill one analysis window -- so that is all the tests override.
# It is set explicitly rather than left at the default so a change to the default
# would show up here as a test failure rather than as a silent shift in meaning.
AUDIO = replace(AudioConfig(), min_speech_seconds=1.0)


def tone(seconds: float, *, frequency: float = 220.0, amplitude: float = 0.3) -> np.ndarray:
    """A steady tone: the VAD accepts it as speech and quality accepts it as clean."""
    count = int(SAMPLE_RATE * seconds)
    timeline = np.arange(count, dtype=np.float32) / SAMPLE_RATE
    return amplitude * np.sin(2.0 * math.pi * frequency * timeline).astype(np.float32)


def speech_then_silence(seconds: float, speech_seconds: float) -> np.ndarray:
    """A file with a controlled amount of speech followed by digital silence."""
    loud = tone(speech_seconds)
    quiet = np.zeros(int(SAMPLE_RATE * (seconds - speech_seconds)), dtype=np.float32)
    return np.concatenate([loud, quiet])


def write_wav(root: Path, relative: str, samples: np.ndarray) -> Path:
    """Write mono float samples as a WAV and return the path."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, SAMPLE_RATE, subtype="PCM_16")
    return path


def record(
    sample_id: str,
    audio_path: str,
    *,
    dataset_id: str = "corpus",
    label: str = BONA_FIDE,
    **metadata: object,
) -> SourceRecord:
    """Build a source record with only the fields validation cares about.

    The metadata values are typed ``object`` because they are forwarded to
    ``SourceRecord``'s optional provenance fields, which are deliberately
    heterogeneous (``str``/``int``/``float``/``Mapping``). Typing them as
    ``str`` here would make the helper unable to pass a test's ``None`` or a
    numeric provenance value, and typing them as the union would put the whole
    record's field list in every call site.
    """
    return SourceRecord(
        sample_id=sample_id,
        dataset_id=dataset_id,
        audio_path=audio_path,
        label=label,
        **cast("Any", metadata),
    )


def clean_quality(duration: float) -> QualityReport:
    """A quality report with no issues, so a check under test is the only one failing."""
    return QualityReport(
        sample_rate_hz=SAMPLE_RATE,
        sample_count=int(duration * SAMPLE_RATE),
        duration_seconds=duration,
        peak_amplitude=0.3,
        rms_amplitude=0.2,
        rms_dbfs=-14.0,
        crest_factor_db=3.0,
        dc_offset=0.0,
        silence_ratio=0.0,
        clipping_ratio=0.0,
        non_finite_ratio=0.0,
        snr_db=30.0,
        speech_frame_ratio=1.0,
        issues=(),
        thresholds=AUDIO.quality,
    )


def measured(
    *,
    duration: float = 3.0,
    speech: float = 2.0,
    file_bytes: int = 4_096,
    issues: tuple[str, ...] = (),
) -> FileMeasurement:
    """A measurement built by hand, so one check can be tested without an encode."""
    return FileMeasurement(
        duration_seconds=duration,
        speech_seconds=speech,
        speech_ratio=speech / duration,
        sample_rate=SAMPLE_RATE,
        file_bytes=file_bytes,
        quality=replace(clean_quality(duration), issues=issues),
    )


def check(
    *,
    config: ValidationConfig | None = None,
    audio_config: AudioConfig | None = None,
    duration: float = 3.0,
    speech: float = 2.0,
    file_bytes: int = 4_096,
    issues: tuple[str, ...] = (),
    **metadata: str,
) -> Rejection | None:
    """Validate a hand-built measurement and return the rejection, if any."""
    return validate_measurement(
        record("s1", "a.wav", **metadata),
        measured(duration=duration, speech=speech, file_bytes=file_bytes, issues=issues),
        config or ValidationConfig(),
        audio_config or AUDIO,
    )


def dataset_entry(dataset_id: str = "corpus") -> DatasetEntry:
    """A minimal enabled registry entry, for exercising the build config path."""
    return DatasetEntry(
        dataset_id=dataset_id,
        name="Corpus",
        source="local",
        version="1",
        license="unknown",
        license_status="UNKNOWN",
        task="spoof_detection",
        enabled=True,
        path="raw",
        adapter="generic",
    )


class TestRejection:
    def test_refuses_a_code_outside_the_vocabulary(self) -> None:
        """A code invented at a call site is exactly what the vocabulary prevents."""
        with pytest.raises(DatasetBuildError, match="not one of"):
            Rejection(
                sample_id="s1",
                dataset_id="corpus",
                audio_path="a.wav",
                code="MADE_UP",
                reason="because",
            )

    def test_to_dict_is_serialisable(self) -> None:
        rejection = Rejection(
            sample_id="s1",
            dataset_id="corpus",
            audio_path="a.wav",
            code=REJECT_TOO_SHORT,
            reason="duration 0.10s is below validation.min_duration_seconds of 0.30s",
        )
        assert rejection.to_dict() == {
            "sample_id": "s1",
            "dataset_id": "corpus",
            "audio_path": "a.wav",
            "code": REJECT_TOO_SHORT,
            "reason": "duration 0.10s is below validation.min_duration_seconds of 0.30s",
        }

    def test_the_vocabulary_contains_no_label_code(self) -> None:
        """The schema refuses an invalid label at construction, so no build can emit it."""
        assert "MISSING_LABEL" not in REJECTION_CODES


class TestCheckPrecedence:
    """One reason per file, and the most fundamental one.

    A file can fail every check at once. Reporting whichever failure was evaluated
    last would send an operator to a threshold that was never the constraint.
    """

    def test_metadata_is_reported_before_size(self) -> None:
        config = ValidationConfig(require_metadata=("label", "speaker_id"), max_file_bytes=1_024)
        rejection = check(config=config, file_bytes=9_999, speaker_id=UNKNOWN)
        assert rejection is not None
        assert rejection.code == REJECT_MISSING_METADATA

    def test_size_is_reported_before_duration(self) -> None:
        config = ValidationConfig(max_file_bytes=1_024)
        rejection = check(config=config, duration=0.1, file_bytes=9_999)
        assert rejection is not None
        assert rejection.code == REJECT_FILE_TOO_LARGE

    def test_duration_is_reported_before_speech(self) -> None:
        """A file below the minimum is rejected on duration whatever it contains."""
        rejection = check(duration=0.1, speech=0.0)
        assert rejection is not None
        assert rejection.code == REJECT_TOO_SHORT

    def test_unusable_signal_is_reported_before_speech(self, tmp_path: Path) -> None:
        """'This recording is silence' is more fundamental than 'too little speech'."""
        path = write_wav(tmp_path, "quiet.wav", speech_then_silence(3.0, 0.05))
        rejection = validate_measurement(
            record("s1", "a.wav"),
            measure_file(path, AUDIO),
            ValidationConfig(),
            AUDIO,
        )
        assert rejection is not None
        assert rejection.code == REJECT_UNUSABLE_SIGNAL

    def test_accepts_a_clean_measurement(self) -> None:
        assert check() is None


class TestIndividualChecks:
    def test_rejects_a_short_file_with_both_figures(self) -> None:
        rejection = check(duration=0.1, speech=0.1)
        assert rejection is not None
        assert rejection.code == REJECT_TOO_SHORT
        assert "0.10s" in rejection.reason
        assert "0.30s" in rejection.reason

    def test_rejects_a_long_file_with_both_figures(self) -> None:
        config = ValidationConfig(min_duration_seconds=0.5, max_duration_seconds=1.0)
        rejection = check(config=config, duration=4.0, speech=4.0)
        assert rejection is not None
        assert rejection.code == REJECT_TOO_LONG
        assert "4.00s" in rejection.reason
        assert "1.00s" in rejection.reason

    def test_rejects_too_little_speech_with_both_figures(self) -> None:
        rejection = check(speech=0.05)
        assert rejection is not None
        assert rejection.code == REJECT_NO_SPEECH
        assert "0.05s" in rejection.reason
        assert "1.00s" in rejection.reason

    def test_speech_check_can_be_turned_off(self) -> None:
        assert check(config=ValidationConfig(require_speech=False), speech=0.0) is None

    def test_speech_threshold_is_the_phase_one_window(self) -> None:
        """The floor is Phase 1's, so one definition of 'enough speech' is used."""
        audio = replace(AUDIO, min_speech_seconds=2.0)
        rejection = check(audio_config=audio, duration=4.0, speech=1.5)
        assert rejection is not None
        assert rejection.code == REJECT_NO_SPEECH
        assert "2.00s" in rejection.reason

    def test_rejects_unknown_required_metadata_naming_the_field(self) -> None:
        config = ValidationConfig(require_metadata=("label", "generator_id"))
        rejection = check(config=config, generator_id=UNKNOWN)
        assert rejection is not None
        assert rejection.code == REJECT_MISSING_METADATA
        assert "generator_id" in rejection.reason

    def test_accepts_published_metadata(self) -> None:
        config = ValidationConfig(require_metadata=("label", "generator_id"))
        assert check(config=config, generator_id="vocoder-1") is None

    def test_requirement_naming_no_field_is_a_config_error(self) -> None:
        """A typo would otherwise be satisfied by every record and go unreported."""
        config = ValidationConfig(require_metadata=("speakerid",))
        with pytest.raises(DatasetConfigError, match="speakerid"):
            validate_sources([], Path("."), config, audio_config=AUDIO)

    def test_rejects_warnings_when_configured(self) -> None:
        config = ValidationConfig(reject_warning_issues=True)
        rejection = check(config=config, issues=(ISSUE_CLIPPED,))
        assert rejection is not None
        assert rejection.code == REJECT_QUALITY_WARNING
        assert ISSUE_CLIPPED in rejection.reason

    def test_tolerates_warnings_by_default(self) -> None:
        assert check(issues=(ISSUE_CLIPPED,)) is None

    def test_a_blocking_issue_is_refused_even_with_warnings_tolerated(self) -> None:
        rejection = check(issues=(ISSUE_SILENT, ISSUE_CLIPPED))
        assert rejection is not None
        assert rejection.code == REJECT_UNUSABLE_SIGNAL
        assert ISSUE_SILENT in rejection.reason
        assert ISSUE_CLIPPED not in rejection.reason


class TestValidateSources:
    def test_accepts_a_usable_file(self, tmp_path: Path) -> None:
        write_wav(tmp_path, "raw/a.wav", speech_then_silence(3.0, 2.0))
        result = validate_sources([record("s1", "raw/a.wav")], tmp_path, audio_config=AUDIO)
        assert result.accepted_ids() == frozenset({"s1"})
        assert result.rejected == ()
        assert result.stats.accepted == 1
        assert result.stats.duration_seconds == pytest.approx(3.0, abs=0.1)

    def test_keeps_every_file_accounted_for(self, tmp_path: Path) -> None:
        """Accepted plus rejected must equal the input, or files vanish silently."""
        write_wav(tmp_path, "raw/good.wav", speech_then_silence(3.0, 2.0))
        broken = tmp_path / "raw/broken.wav"
        broken.parent.mkdir(parents=True, exist_ok=True)
        broken.write_bytes(b"RIFF\x00\x00\x00\x00WAVE not audio")
        result = validate_sources(
            [
                record("good", "raw/good.wav"),
                record("broken", "raw/broken.wav"),
                record("absent", "raw/absent.wav"),
            ],
            tmp_path,
            audio_config=AUDIO,
        )
        assert result.accepted_ids() | result.rejected_ids() == {
            "good",
            "broken",
            "absent",
        }
        assert not result.accepted_ids() & result.rejected_ids()
        assert result.stats.accepted + result.stats.rejected == result.stats.total

    def test_reports_a_missing_file_as_unreadable(self, tmp_path: Path) -> None:
        result = validate_sources([record("s1", "raw/absent.wav")], tmp_path, audio_config=AUDIO)
        rejection = result.rejected[0]
        assert rejection.code == REJECT_UNREADABLE
        assert "cannot stat" in rejection.reason

    def test_reports_undecodable_bytes(self, tmp_path: Path) -> None:
        broken = tmp_path / "raw/broken.wav"
        broken.parent.mkdir(parents=True, exist_ok=True)
        broken.write_bytes(b"RIFF\x00\x00\x00\x00WAVE not audio at all")
        result = validate_sources([record("s1", "raw/broken.wav")], tmp_path, audio_config=AUDIO)
        assert result.rejected[0].code in {REJECT_UNDECODABLE, REJECT_UNSUPPORTED_FORMAT}

    def test_reports_digital_silence(self, tmp_path: Path) -> None:
        write_wav(tmp_path, "raw/silent.wav", np.zeros(SAMPLE_RATE * 2, dtype=np.float32))
        result = validate_sources([record("s1", "raw/silent.wav")], tmp_path, audio_config=AUDIO)
        assert result.rejected[0].code == REJECT_UNUSABLE_SIGNAL

    def test_reports_a_file_over_the_size_limit(self, tmp_path: Path) -> None:
        write_wav(tmp_path, "raw/big.wav", speech_then_silence(3.0, 2.0))
        config = ValidationConfig(max_file_bytes=1_024)
        result = validate_sources(
            [record("s1", "raw/big.wav")], tmp_path, config, audio_config=AUDIO
        )
        rejection = result.rejected[0]
        assert rejection.code == REJECT_FILE_TOO_LARGE
        assert "max_file_bytes" in rejection.reason

    def test_does_not_truncate_the_inventory_at_the_size_limit(self, tmp_path: Path) -> None:
        """Forwarding the cap to discovery would report oversize as merely unreadable."""
        write_wav(tmp_path, "raw/big.wav", speech_then_silence(3.0, 2.0))
        entries, _unreadable = build_inventory([record("s1", "raw/big.wav")], tmp_path)
        assert len(entries) == 1
        result = validate_sources(
            [record("s1", "raw/big.wav")],
            tmp_path,
            ValidationConfig(max_file_bytes=1_024),
            audio_config=AUDIO,
        )
        assert result.rejected[0].code == REJECT_FILE_TOO_LARGE

    def test_reports_an_intake_budget_overrun(self, tmp_path: Path) -> None:
        """A long file trips Phase 1's intake limit before the policy's own check."""
        write_wav(tmp_path, "raw/long.wav", speech_then_silence(20.0, 20.0))
        audio = replace(AUDIO, max_duration_seconds=5.0)
        result = validate_sources(
            [record("s1", "raw/long.wav")], tmp_path, ValidationConfig(), audio_config=audio
        )
        assert result.rejected[0].code == REJECT_AUDIO_BUDGET

    def test_reuses_a_discovery_result(self, tmp_path: Path) -> None:
        """A build must not hash a corpus twice, and must keep discovery's reasons."""
        write_wav(tmp_path, "raw/a.wav", speech_then_silence(3.0, 2.0))
        records = [record("gone", "raw/gone.wav")]
        discovery = discover(records, tmp_path)
        result = validate_sources(records, tmp_path, audio_config=AUDIO, inventory=discovery)
        assert result.rejected[0].code == REJECT_UNREADABLE
        assert "cannot stat" in result.rejected[0].reason

    def test_reuses_a_plain_inventory(self, tmp_path: Path) -> None:
        write_wav(tmp_path, "raw/a.wav", speech_then_silence(3.0, 2.0))
        records = [record("a", "raw/a.wav")]
        entries, _unreadable = build_inventory(records, tmp_path)
        result = validate_sources(records, tmp_path, audio_config=AUDIO, inventory=entries)
        assert result.accepted_ids() == frozenset({"a"})

    def test_rejects_duplicate_sample_ids(self, tmp_path: Path) -> None:
        write_wav(tmp_path, "raw/a.wav", speech_then_silence(3.0, 2.0))
        records = [record("s1", "raw/a.wav"), record("s1", "raw/a.wav")]
        with pytest.raises(DatasetBuildError, match="two records with sample_id"):
            validate_sources(records, tmp_path, audio_config=AUDIO)

    def test_orders_results_deterministically(self, tmp_path: Path) -> None:
        write_wav(tmp_path, "raw/a.wav", speech_then_silence(3.0, 2.0))
        write_wav(tmp_path, "raw/b.wav", speech_then_silence(0.2, 0.2))
        write_wav(tmp_path, "raw/c.wav", speech_then_silence(0.2, 0.2))
        records = [
            record("z", "raw/a.wav"),
            record("b", "raw/b.wav"),
            record("a", "raw/c.wav"),
        ]
        first = validate_sources(records, tmp_path, audio_config=AUDIO)
        second = validate_sources(list(reversed(records)), tmp_path, audio_config=AUDIO)
        assert [item.sample_id for item in first.accepted] == ["z"]
        assert [item.sample_id for item in first.rejected] == ["a", "b"]
        assert first.rejected == second.rejected
        assert first.accepted == second.accepted

    def test_groups_reasons_for_the_report(self, tmp_path: Path) -> None:
        write_wav(tmp_path, "raw/a.wav", speech_then_silence(0.2, 0.2))
        write_wav(tmp_path, "raw/b.wav", speech_then_silence(0.2, 0.2))
        result = validate_sources(
            [record("a", "raw/a.wav"), record("b", "raw/b.wav")], tmp_path, audio_config=AUDIO
        )
        grouped = result.rejections_by_code()
        assert set(grouped) == {REJECT_TOO_SHORT}
        assert len(grouped[REJECT_TOO_SHORT]) == 2
        assert result.stats.rejected_per_reason == {REJECT_TOO_SHORT: 2}

    def test_totals_match_the_two_sets(self, tmp_path: Path) -> None:
        write_wav(tmp_path, "raw/a.wav", speech_then_silence(3.0, 2.0))
        write_wav(tmp_path, "raw/b.wav", speech_then_silence(0.2, 0.2))
        result = validate_sources(
            [
                record("a", "raw/a.wav", dataset_id="one"),
                record("b", "raw/b.wav", dataset_id="two"),
            ],
            tmp_path,
            audio_config=AUDIO,
        )
        assert result.stats.total == 2
        assert result.stats.accepted_per_dataset == {"one": 1}
        assert sum(result.stats.rejected_per_reason.values()) == result.stats.rejected

    def test_returns_records_for_the_splitting_stage(self, tmp_path: Path) -> None:
        write_wav(tmp_path, "raw/a.wav", speech_then_silence(3.0, 2.0))
        result = validate_sources([record("a", "raw/a.wav")], tmp_path, audio_config=AUDIO)
        assert [item.sample_id for item in result.accepted_records()] == ["a"]

    def test_every_emitted_code_is_in_the_vocabulary(self, tmp_path: Path) -> None:
        write_wav(tmp_path, "raw/a.wav", speech_then_silence(3.0, 2.0))
        write_wav(tmp_path, "raw/short.wav", speech_then_silence(0.2, 0.2))
        write_wav(tmp_path, "raw/silent.wav", np.zeros(SAMPLE_RATE * 2, dtype=np.float32))
        records = [
            record("a", "raw/a.wav"),
            record("short", "raw/short.wav"),
            record("silent", "raw/silent.wav"),
            record("absent", "raw/absent.wav"),
        ]
        result = validate_sources(records, tmp_path, audio_config=AUDIO)
        assert {item.code for item in result.rejected} == {
            REJECT_TOO_SHORT,
            REJECT_UNUSABLE_SIGNAL,
            REJECT_UNREADABLE,
        }
        for rejection in result.rejected:
            assert rejection.code in REJECTION_CODES
            assert rejection.reason.strip()

    def test_uses_default_configuration(self, tmp_path: Path) -> None:
        write_wav(tmp_path, "raw/a.wav", speech_then_silence(3.0, 2.0))
        assert validate_sources([record("a", "raw/a.wav")], tmp_path).accepted_ids() == {"a"}

    def test_accepts_an_absolute_audio_path(self, tmp_path: Path) -> None:
        """A corpus on a separate volume is a real deployment, not an error."""
        path = write_wav(tmp_path, "elsewhere/a.wav", speech_then_silence(3.0, 2.0))
        result = validate_sources([record("a", str(path))], tmp_path, audio_config=AUDIO)
        assert result.accepted_ids() == frozenset({"a"})


class TestDuplicates:
    def test_drops_a_byte_identical_file_and_names_the_kept_one(self, tmp_path: Path) -> None:
        samples = speech_then_silence(3.0, 2.0)
        write_wav(tmp_path, "raw/a.wav", samples)
        write_wav(tmp_path, "raw/copy.wav", samples)
        result = validate_sources(
            [record("a", "raw/a.wav"), record("copy", "raw/copy.wav")],
            tmp_path,
            ValidationConfig(dedup_scope="file"),
            audio_config=AUDIO,
        )
        assert result.accepted_ids() == frozenset({"a"})
        rejection = result.rejected[0]
        assert rejection.code == REJECT_DUPLICATE_FILE
        assert "'a'" in rejection.reason

    def test_keeps_the_first_id_in_a_three_way_tie(self, tmp_path: Path) -> None:
        samples = speech_then_silence(3.0, 2.0)
        for name in ("a", "b", "c"):
            write_wav(tmp_path, f"raw/{name}.wav", samples)
        records = [record(name, f"raw/{name}.wav") for name in ("a", "b", "c")]
        result = validate_sources(
            records, tmp_path, ValidationConfig(dedup_scope="file"), audio_config=AUDIO
        )
        assert result.accepted_ids() == frozenset({"a"})
        assert len(result.rejected) == 2

    def test_a_refused_copy_does_not_consume_the_kept_slot(self, tmp_path: Path) -> None:
        """A duplicate of a file that is itself refused must not lose the good copy."""
        samples = speech_then_silence(0.2, 0.2)
        write_wav(tmp_path, "raw/short.wav", samples)
        write_wav(tmp_path, "raw/copy.wav", samples)
        write_wav(tmp_path, "raw/good.wav", speech_then_silence(3.0, 2.0))
        records = [
            record("short", "raw/short.wav"),
            record("copy", "raw/copy.wav"),
            record("good", "raw/good.wav"),
        ]
        result = validate_sources(
            records, tmp_path, ValidationConfig(dedup_scope="file"), audio_config=AUDIO
        )
        assert result.accepted_ids() == frozenset({"good"})
        assert {item.code for item in result.rejected} == {REJECT_TOO_SHORT}

    def test_can_be_turned_off(self, tmp_path: Path) -> None:
        samples = speech_then_silence(3.0, 2.0)
        write_wav(tmp_path, "raw/a.wav", samples)
        write_wav(tmp_path, "raw/copy.wav", samples)
        records = [record("a", "raw/a.wav"), record("copy", "raw/copy.wav")]
        result = validate_sources(
            records,
            tmp_path,
            ValidationConfig(reject_duplicates=False, dedup_scope="file"),
            audio_config=AUDIO,
        )
        assert len(result.accepted) == 2
        assert result.stats.unavailable_scopes == ()

    def test_drops_identical_stored_audio(self, tmp_path: Path) -> None:
        """Two encodings of the same audio share a stored-content hash."""
        write_wav(tmp_path, "raw/a.wav", speech_then_silence(3.0, 2.0))
        write_wav(tmp_path, "raw/b.wav", speech_then_silence(3.0, 2.0))
        result = validate_sources(
            [record("a", "raw/a.wav"), record("b", "raw/b.wav")],
            tmp_path,
            ValidationConfig(dedup_scope="content"),
            audio_config=AUDIO,
            content_hashes={"a": "sha256:1111", "b": "sha256:1111"},
        )
        assert result.accepted_ids() == frozenset({"a"})
        assert result.rejected[0].code == REJECT_DUPLICATE_CONTENT

    def test_reports_content_scope_as_unavailable_without_hashes(self, tmp_path: Path) -> None:
        """A scope with nothing to compare has not passed; it is reported unchecked."""
        write_wav(tmp_path, "raw/a.wav", speech_then_silence(3.0, 2.0))
        result = validate_sources(
            [record("a", "raw/a.wav")],
            tmp_path,
            ValidationConfig(dedup_scope="content"),
            audio_config=AUDIO,
        )
        assert result.stats.unavailable_scopes == ("content",)
        assert result.accepted_ids() == frozenset({"a"})

    def test_content_scope_leaves_file_duplicates_alone(self, tmp_path: Path) -> None:
        samples = speech_then_silence(3.0, 2.0)
        write_wav(tmp_path, "raw/a.wav", samples)
        write_wav(tmp_path, "raw/copy.wav", samples)
        result = validate_sources(
            [record("a", "raw/a.wav"), record("copy", "raw/copy.wav")],
            tmp_path,
            ValidationConfig(dedup_scope="content"),
            audio_config=AUDIO,
            content_hashes={"a": "sha256:1111", "copy": "sha256:2222"},
        )
        assert len(result.accepted) == 2
        assert result.stats.unavailable_scopes == ()


class TestAcceptance:
    def test_exposes_the_figures_a_manifest_needs(self, tmp_path: Path) -> None:
        write_wav(tmp_path, "raw/a.wav", speech_then_silence(3.0, 2.0))
        result = validate_sources([record("a", "raw/a.wav")], tmp_path, audio_config=AUDIO)
        item: Acceptance = result.accepted[0]
        assert item.sample_id == "a"
        assert item.dataset_id == "corpus"
        assert item.duration_seconds == pytest.approx(3.0, abs=0.1)
        assert item.speech_seconds > 0.0
        assert item.warnings == ()
        assert item.record.sample_id == "a"

    def test_retains_no_audio(self, tmp_path: Path) -> None:
        """A result covering a corpus must not become a second copy of it."""
        write_wav(tmp_path, "raw/a.wav", speech_then_silence(3.0, 2.0))
        result = validate_sources([record("a", "raw/a.wav")], tmp_path, audio_config=AUDIO)
        measurement = result.accepted[0].measurement
        retained = {field.name: getattr(measurement, field.name) for field in fields(measurement)}
        assert set(retained) == {
            "duration_seconds",
            "speech_seconds",
            "speech_ratio",
            "sample_rate",
            "file_bytes",
            "quality",
        }
        assert not any(isinstance(value, np.ndarray) for value in retained.values()), (
            "a decoded array survived into the validation result"
        )


class TestEmptyCorpus:
    def test_an_empty_corpus_validates(self, tmp_path: Path) -> None:
        result = validate_sources([], tmp_path, audio_config=AUDIO)
        assert result == ValidationResult()
        assert result.stats.total == 0
        assert result.accepted == ()


class TestBuildIntegration:
    def test_uses_the_datasets_own_audio_config(self, tmp_path: Path) -> None:
        """``DataConfig.audio_config`` is the Phase 1 config a build should pass."""
        write_wav(tmp_path, "raw/a.wav", speech_then_silence(3.0, 2.0))
        data_config = DataConfig(root=str(tmp_path), datasets=(dataset_entry(),))
        audio = data_config.audio_config()
        result = validate_sources(
            [record("a", "raw/a.wav")], tmp_path, data_config.validation, audio_config=audio
        )
        assert result.accepted_ids() == frozenset({"a"})
        assert audio.min_speech_seconds == audio.min_segment_seconds

    def test_a_dataset_window_shorter_than_the_speech_floor_reports_speech(
        self, tmp_path: Path
    ) -> None:
        """The speech floor scales with the window, so a narrowed one is not self-contradictory.

        0.8s of speech is well under Phase 1's 2.0s absolute default yet clears the
        0.5s floor a 1.0s window implies. Accepted here means the floor moved with
        the window; a fixed 2.0s floor would have refused the file.
        """
        write_wav(tmp_path, "raw/a.wav", speech_then_silence(3.0, 0.8))
        data_config = DataConfig(
            root=str(tmp_path), window_seconds=1.0, datasets=(dataset_entry(),)
        )
        audio = data_config.audio_config()
        assert audio.min_speech_seconds == 0.5
        result = validate_sources(
            [record("a", "raw/a.wav")], tmp_path, data_config.validation, audio_config=audio
        )
        assert result.accepted_ids() == frozenset({"a"})
