"""Unit tests for leakage detection.

Leakage checks answer one question: does a *known* identity unit appear on more
than one split side? The tests therefore concentrate on the two ways the check
can lie -- counting an unknown value as an identity (it never is) and missing a
unit the check should have seen (two files of one recording, the same bytes
under two names). Each axis is exercised once so a regression is attributable.
"""

from __future__ import annotations

import json

import pytest

from voxshield.data.errors import DatasetConfigError, DatasetLeakageError
from voxshield.data.labels import encode_label
from voxshield.data.leakage import (
    AXIS_CHANNEL,
    AXIS_DEVICE,
    AXIS_FILE,
    AXIS_PARENT,
    AXIS_SPEAKER,
    LEAKAGE_AXES,
    assert_no_leakage,
    check_leakage,
)
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
    content_hash: str = "",
) -> SampleRecord:
    """A segment record with only the identity fields a test changes."""
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
        content_hash=content_hash,
        source_split=UNKNOWN,
        preprocessing_version="phase1.0",
        dataset_build_id="",
    )


def source_record(sample_id: str, *, parent_id: str, speaker_id: str = UNKNOWN) -> SourceRecord:
    """A source record, split carried by an explicit mapping."""
    return SourceRecord(
        sample_id=sample_id,
        dataset_id="corpus",
        audio_path=f"raw/{sample_id}.wav",
        label=BONA_FIDE,
        parent_id=parent_id,
        speaker_id=speaker_id,
    )


class TestNoLeakage:
    """A disjoint corpus makes every axis claimable and clean."""

    def test_disjoint_records_are_clean(self) -> None:
        records = (
            sample("t1", split="train", parent_id="p1", speaker_id="spk1"),
            sample("t2", split="train", parent_id="p2", speaker_id="spk2"),
            sample("e1", split="test", parent_id="p3", speaker_id="spk3"),
        )

        report = check_leakage(records)

        assert report.n_records == 3
        assert not report.has_leakage
        assert report.leaked_axes == ()
        assert report.check_for(AXIS_PARENT).passed
        assert report.check_for(AXIS_SPEAKER).available


class TestLeakDetection:
    """Each unit the check must see, seen."""

    def test_parent_leak(self) -> None:
        records = (
            sample("t1", split="train", parent_id="p1", speaker_id="spk1"),
            sample("e1", split="test", parent_id="p1", speaker_id="spk2"),
        )

        report = check_leakage(records)

        assert report.has_leakage
        assert report.check_for(AXIS_PARENT).leaked
        assert not report.check_for(AXIS_SPEAKER).leaked

    def test_speaker_leak(self) -> None:
        records = (
            sample("t1", split="train", parent_id="p1", speaker_id="spk1"),
            sample("e1", split="test", parent_id="p2", speaker_id="spk1"),
        )

        report = check_leakage(records)

        assert report.check_for(AXIS_SPEAKER).leaked

    def test_file_leak_via_source_file_hash(self) -> None:
        records = (
            sample("t1", split="train", parent_id="p1", file_hash="sha256:dup"),
            sample("e1", split="test", parent_id="p2", file_hash="sha256:dup"),
        )

        report = check_leakage(records)

        assert report.check_for(AXIS_FILE).leaked

    def test_content_leak_on_reencoded_segments(self) -> None:
        records = (
            sample(
                "t1",
                split="train",
                parent_id="p1",
                file_hash="sha256:a",
                content_hash="sha256:audio",
            ),
            sample(
                "e1",
                split="test",
                parent_id="p2",
                file_hash="sha256:b",
                content_hash="sha256:audio",
            ),
        )

        report = check_leakage(records)

        assert report.check_for(AXIS_FILE).leaked
        assert "sha256:audio" in report.check_for(AXIS_FILE).detail

    def test_file_leak_via_file_hashes_mapping_for_sources(self) -> None:
        records = (
            source_record("t1", parent_id="p1"),
            source_record("e1", parent_id="p2"),
        )
        splits = {"t1": "train", "e1": "test"}

        report = check_leakage(
            records, splits=splits, file_hashes={"t1": "sha256:dup", "e1": "sha256:dup"}
        )

        assert report.check_for(AXIS_FILE).leaked

    def test_unknown_identity_is_never_a_leak(self) -> None:
        records = (
            sample("t1", split="train", parent_id="p1", speaker_id=UNKNOWN),
            sample("e1", split="test", parent_id="p2", speaker_id=UNKNOWN),
        )

        report = check_leakage(records)

        speaker = report.check_for(AXIS_SPEAKER)
        assert not speaker.leaked
        assert not speaker.available


class TestPopulationAxes:
    """Channel and device are population axes, checked the same way.

    They differ from speaker in what a clean result *means*: a corpus with two
    channel populations can be perfectly disjoint, and one unknown channel value
    makes the corpus unable to claim disjointness without leaking anything.
    """

    def test_channel_leak(self) -> None:
        records = (
            sample("t1", split="train", parent_id="p1", channel="narrowband"),
            sample("e1", split="test", parent_id="p2", channel="narrowband"),
        )

        report = check_leakage(records)

        check = report.check_for(AXIS_CHANNEL)
        assert check.leaked
        assert "narrowband" in check.detail
        # A speaker-disjoint corpus is still channel-leaky, which is exactly why
        # the axis needs its own check rather than riding on the speaker axis.
        assert not report.check_for(AXIS_SPEAKER).leaked

    def test_device_leak(self) -> None:
        records = (
            sample("t1", split="train", parent_id="p1", device="handsetA"),
            sample("e1", split="test", parent_id="p2", device="handsetA"),
        )

        report = check_leakage(records)

        assert report.check_for(AXIS_DEVICE).leaked

    def test_populations_pinned_to_one_side_are_clean(self) -> None:
        records = (
            sample("t1", split="train", parent_id="p1", channel="narrowband", device="devA"),
            sample("t2", split="train", parent_id="p2", channel="narrowband", device="devA"),
            sample("e1", split="test", parent_id="p3", channel="wideband", device="devB"),
        )

        report = check_leakage(records)

        assert report.check_for(AXIS_CHANNEL).passed
        assert report.check_for(AXIS_DEVICE).passed
        assert report.check_for(AXIS_CHANNEL).known_values == 2

    def test_unknown_population_is_neither_leak_nor_evidence(self) -> None:
        records = (
            sample("t1", split="train", parent_id="p1", channel=UNKNOWN, device=UNKNOWN),
            sample("e1", split="test", parent_id="p2", channel=UNKNOWN, device=UNKNOWN),
        )

        report = check_leakage(records)

        for axis in (AXIS_CHANNEL, AXIS_DEVICE):
            check = report.check_for(axis)
            assert not check.leaked
            assert not check.available

    def test_an_unknown_value_does_not_hide_a_real_leak(self) -> None:
        records = (
            sample("t1", split="train", parent_id="p1", channel="wideband"),
            sample("e2", split="train", parent_id="p2", channel=UNKNOWN),
            sample("e1", split="test", parent_id="p3", channel="wideband"),
        )

        report = check_leakage(records)

        assert report.check_for(AXIS_CHANNEL).leaked

    def test_a_leaked_population_blocks_when_required(self) -> None:
        report = check_leakage(
            (
                sample("t1", split="train", parent_id="p1", channel="narrowband"),
                sample("e1", split="test", parent_id="p2", channel="narrowband"),
            )
        )

        with pytest.raises(DatasetLeakageError, match="channel"):
            assert_no_leakage(report, required=(AXIS_CHANNEL,), fail_on_unavailable=True)

    def test_an_unpublished_population_blocks_when_required(self) -> None:
        report = check_leakage(
            (
                sample("t1", split="train", parent_id="p1", channel=UNKNOWN),
                sample("e1", split="test", parent_id="p2", channel=UNKNOWN),
            )
        )

        with pytest.raises(DatasetLeakageError, match="channel"):
            assert_no_leakage(report, required=(AXIS_CHANNEL,), fail_on_unavailable=True)


class TestConfiguration:
    """Policy gates: what blocks, what reports, what the check refuses to do."""

    def test_missing_split_is_a_config_error(self) -> None:
        records = (
            source_record("t1", parent_id="p1"),
            source_record("e1", parent_id="p2"),
        )

        with pytest.raises(DatasetConfigError):
            check_leakage(records)

    def test_assert_blocks_a_mandatory_axis_leak(self) -> None:
        report = check_leakage(
            (
                sample("t1", split="train", parent_id="p1"),
                sample("e1", split="test", parent_id="p1"),
            )
        )

        with pytest.raises(DatasetLeakageError):
            assert_no_leakage(report, required=(AXIS_PARENT,))

    def test_optional_axis_leak_is_report_only(self) -> None:
        report = check_leakage(
            (
                sample("t1", split="train", parent_id="p1", generator_id="g1"),
                sample("e1", split="test", parent_id="p2", generator_id="g1"),
            )
        )

        # Required axes (parent, file) are clean; generator leaked but is not
        # in the required set.
        assert_no_leakage(report, required=(AXIS_PARENT, AXIS_FILE))

    def test_unevaluable_mandatory_axis_blocks_when_required(self) -> None:
        report = check_leakage(
            (
                sample("t1", split="train", parent_id="p1", speaker_id=UNKNOWN),
                sample("e1", split="test", parent_id="p2", speaker_id=UNKNOWN),
            )
        )

        with pytest.raises(DatasetLeakageError):
            assert_no_leakage(report, required=(AXIS_SPEAKER,), fail_on_unavailable=True)

        # Without the flag the same report passes -- the corpus merely cannot
        # make a speaker-disjoint claim.
        assert_no_leakage(report, required=(AXIS_SPEAKER,))

    def test_check_for_unknown_axis_returns_none(self) -> None:
        report = check_leakage((sample("t1", split="train", parent_id="p1"),))

        assert report.check_for("breathiness") is None


class TestReport:
    """Leakage reports are JSON-serialisable for a build report."""

    def test_report_roundtrips_through_json(self) -> None:
        report = check_leakage(
            (
                sample("t1", split="train", parent_id="p1"),
                sample("e1", split="test", parent_id="p1"),
            )
        )

        payload = json.loads(json.dumps(report.to_dict()))

        assert payload["n_records"] == 2
        assert payload["has_leakage"] is True
        assert set(LEAKAGE_AXES) <= {check["axis"] for check in payload["checks"]}
