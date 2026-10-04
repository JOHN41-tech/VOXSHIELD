"""Unit tests for the build-time leakage gates.

The gates answer a go/no-go question the manifest can already answer
analytically: can this corpus honestly claim disjoint train/test sets? The
strictness lives in configuration, so the tests also pin the two ways a gate
policy can be under- or over-strict -- an optional axis that leaked but was
configured acceptable, and a mandatory axis with no known identities that must
refuse the build when the operator demands the claim.
"""

from __future__ import annotations

import pytest

from voxshield.data.config import GateConfig, SplitConfig
from voxshield.data.errors import DatasetConfigError, DatasetLeakageError
from voxshield.data.gates import assert_gates_pass, evaluate_leakage_gates
from voxshield.data.labels import encode_label
from voxshield.data.schema import UNKNOWN, SampleRecord, SourceRecord

BONA_FIDE = "bona_fide"


def sample(
    sample_id: str,
    *,
    split: str,
    parent_id: str,
    speaker_id: str = UNKNOWN,
    generator_id: str = UNKNOWN,
    channel: str = UNKNOWN,
    device: str = UNKNOWN,
    file_hash: str = "",
) -> SampleRecord:
    """A segment record carrying the identity fields the gates read."""
    return SampleRecord(
        sample_id=sample_id,
        dataset_id="corpus",
        audio_path=f"processed/segments/corpus/{sample_id}.wav",
        label=BONA_FIDE,
        label_index=encode_label(BONA_FIDE),
        split=split,
        parent_id=parent_id,
        segment_index=0,
        start_seconds=0.0,
        duration_seconds=4.0,
        sample_rate=16_000,
        speech_seconds=4.0,
        coverage=1.0,
        is_padded=False,
        waveform_samples=64_000,
        speaker_id=speaker_id,
        generator_id=generator_id,
        language=UNKNOWN,
        codec=UNKNOWN,
        session_id=UNKNOWN,
        attack_type=UNKNOWN,
        channel=channel,
        device=device,
        recorded_at=UNKNOWN,
        file_hash=file_hash,
        content_hash="",
        source_split=UNKNOWN,
        preprocessing_version="phase1.0",
        dataset_build_id="",
    )


def disjoint_1() -> tuple[SampleRecord, ...]:
    """Three clean segments: two train, one test."""
    return (
        sample("t1", split="train", parent_id="p1", speaker_id="spk1", file_hash="sha256:a"),
        sample("t2", split="train", parent_id="p2", speaker_id="spk2", file_hash="sha256:b"),
        sample("e1", split="test", parent_id="p3", speaker_id="spk3", file_hash="sha256:c"),
    )


class TestBasic:
    """A clean corpus passes the standard policy with nothing to say."""

    def test_disjoint_corpus_passes(self) -> None:
        report = evaluate_leakage_gates(disjoint_1())

        assert report.passed
        assert report.failures == ()
        assert report.test_sources == 1
        assert report.min_test_sources == 1

    def test_assert_gates_pass_accepts_a_clean_report(self) -> None:
        assert_gates_pass(evaluate_leakage_gates(disjoint_1()))

    def test_empty_record_set_is_a_config_error(self) -> None:
        with pytest.raises(DatasetConfigError):
            evaluate_leakage_gates([])

    def test_a_source_without_a_split_is_a_config_error(self) -> None:
        source = SourceRecord(
            sample_id="t1",
            dataset_id="corpus",
            audio_path="raw/t1.wav",
            label=BONA_FIDE,
        )

        with pytest.raises(DatasetConfigError):
            evaluate_leakage_gates([source])


class TestLeakageFailure:
    """A leaked mandatory axis blocks the build."""

    def test_shared_parent_across_train_and_test_blocks(self) -> None:
        records = (
            sample("t1", split="train", parent_id="p1", speaker_id="spk1", file_hash="sha256:a"),
            sample("e1", split="test", parent_id="p1", speaker_id="spk2", file_hash="sha256:b"),
        )

        report = evaluate_leakage_gates(records)

        assert not report.passed
        assert any("parent" in failure for failure in report.failures)
        with pytest.raises(DatasetLeakageError):
            assert_gates_pass(report)


class TestSplitDerivedAxes:
    """Channel and device become mandatory when the split asked for them."""

    def test_split_requires_channel_disjoint(self) -> None:
        records = (
            sample("t1", split="train", parent_id="p1", channel="wideband", file_hash="sha256:a"),
            sample("e1", split="test", parent_id="p2", channel="wideband", file_hash="sha256:b"),
        )

        report = evaluate_leakage_gates(records, split_config=SplitConfig(require_channel_disjoint=True))

        assert not report.passed
        assert "channel" in report.required_axes

    def test_split_requires_device_disjoint(self) -> None:
        records = (
            sample("t1", split="train", parent_id="p1", device="devA", file_hash="sha256:a"),
            sample("e1", split="test", parent_id="p2", device="devA", file_hash="sha256:b"),
        )

        report = evaluate_leakage_gates(records, split_config=SplitConfig(require_device_disjoint=True))

        assert not report.passed
        assert "device" in report.required_axes

    def test_unevaluable_channel_blocks_when_required(self) -> None:
        records = (
            sample("t1", split="train", parent_id="p1", channel=UNKNOWN, file_hash="sha256:a"),
            sample("e1", split="test", parent_id="p2", channel=UNKNOWN, file_hash="sha256:b"),
        )

        report = evaluate_leakage_gates(
            records,
            split_config=SplitConfig(require_channel_disjoint=True),
            config=GateConfig(fail_on_unavailable=True),
        )

        assert not report.passed
        assert any("channel" in failure for failure in report.failures)


class TestPolicy:
    """Strictness is configuration, and both directions are pinned."""

    def test_optional_generator_leak_is_a_warning_only(self) -> None:
        records = (
            sample(
                "t1",
                split="train",
                parent_id="p1",
                speaker_id="spk1",
                generator_id="g1",
                file_hash="sha256:a",
            ),
            sample(
                "e1",
                split="test",
                parent_id="p2",
                speaker_id="spk2",
                generator_id="g1",
                file_hash="sha256:b",
            ),
        )

        report = evaluate_leakage_gates(records)

        assert report.passed
        assert "generator" not in report.required_axes
        assert any("generator" in warning for warning in report.warnings)

    def test_requiring_generator_makes_its_leak_block(self) -> None:
        records = (
            sample(
                "t1",
                split="train",
                parent_id="p1",
                speaker_id="spk1",
                generator_id="g1",
                file_hash="sha256:a",
            ),
            sample(
                "e1",
                split="test",
                parent_id="p2",
                speaker_id="spk2",
                generator_id="g1",
                file_hash="sha256:b",
            ),
        )
        config = GateConfig(require_generator_disjoint=True)

        report = evaluate_leakage_gates(records, config=config)

        assert not report.passed
        with pytest.raises(DatasetLeakageError):
            assert_gates_pass(report, config=config)

    def test_min_test_sources_floor_is_a_config_error(self) -> None:
        config = GateConfig(min_test_sources=5)

        with pytest.raises(DatasetConfigError):
            assert_gates_pass(evaluate_leakage_gates(disjoint_1()), config=config)

    def test_unevaluable_mandatory_axis_warns_by_default(self) -> None:
        records = (
            sample("t1", split="train", parent_id="p1", speaker_id=UNKNOWN, file_hash="sha256:a"),
            sample("e1", split="test", parent_id="p2", speaker_id=UNKNOWN, file_hash="sha256:b"),
        )

        report = evaluate_leakage_gates(records)

        assert report.passed
        assert any("weakened" in warning for warning in report.warnings)

    def test_fail_on_unavailable_forces_the_failure(self) -> None:
        records = (
            sample("t1", split="train", parent_id="p1", speaker_id=UNKNOWN, file_hash="sha256:a"),
            sample("e1", split="test", parent_id="p2", speaker_id=UNKNOWN, file_hash="sha256:b"),
        )
        config = GateConfig(fail_on_unavailable=True)

        report = evaluate_leakage_gates(records, config=config)

        assert not report.passed
        assert any("no known identity" in failure for failure in report.failures)
        with pytest.raises(DatasetLeakageError):
            assert_gates_pass(report, config=config)
