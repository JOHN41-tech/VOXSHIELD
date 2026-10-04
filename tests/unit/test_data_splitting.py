"""Unit tests for split assignment.

The property under test is whole-group (speaker-disjoint by construction)
assignment: no known speaker, and no parent recording, may appear on more than
one split side. Secondary properties matter because a split is an artefact an
operator diffes: determinism (same seed, same corpus, same result), stable
reports, and an honest fallback when temporal data cannot support the claim.
"""

from __future__ import annotations

import json

import pytest

from voxshield.data.config import SplitConfig
from voxshield.data.errors import SplitError
from voxshield.data.schema import UNKNOWN, SourceRecord
from voxshield.data.splitting import (
    SPLIT_NAMES,
    TRAIN,
    assign_splits,
    build_split_report,
    write_split_report,
)

BONA_FIDE = "bona_fide"


def source(
    sample_id: str,
    *,
    speaker_id: str = UNKNOWN,
    generator_id: str = UNKNOWN,
    channel: str = UNKNOWN,
    device: str = UNKNOWN,
    language: str = UNKNOWN,
    recorded_at: str = UNKNOWN,
) -> SourceRecord:
    """A source whose only relevant fields the test names."""
    return SourceRecord(
        sample_id=sample_id,
        dataset_id="corpus",
        audio_path=f"raw/{sample_id}.wav",
        label=BONA_FIDE,
        speaker_id=speaker_id,
        generator_id=generator_id,
        channel=channel,
        device=device,
        language=language,
        recorded_at=recorded_at,
    )


def speakers(count: int) -> list[SourceRecord]:
    """``count`` distinct one-file speakers, for exercising whole-group splits."""
    return [source(f"s{i}", speaker_id=f"spk{i}") for i in range(count)]


class TestWholeGroupAssignment:
    """A split assigns every source exactly once, wholesale per speaker."""

    def test_every_source_lands_in_exactly_one_split(self) -> None:
        assignment = assign_splits(speakers(12))

        assert len(assignment.splits) == 12
        assert sorted(assignment.splits.values()) == sorted(a for a in assignment.splits.values())
        assert set(assignment.splits.values()) <= set(SPLIT_NAMES)

    def test_whole_speakers_stay_whole(self) -> None:
        assignment = assign_splits(speakers(12))

        seen: dict[str, str] = {}
        for record in speakers(12):
            split = assignment.split_for(record.sample_id)
            assert seen.setdefault(record.partition_key, split) == split

    def test_all_splits_are_populated(self) -> None:
        assignment = assign_splits(speakers(12))

        assert all(assignment.counts()[split] > 0 for split in SPLIT_NAMES)

    def test_deterministic_across_repeats(self) -> None:
        first = assign_splits(speakers(12), seed=7)
        second = assign_splits(speakers(12), seed=7)

        assert first.splits == second.splits
        assert first.group_splits == second.group_splits

    def test_a_different_seed_changes_the_assignment(self) -> None:
        first = assign_splits(speakers(30), seed=1)
        second = assign_splits(speakers(30), seed=2)

        assert first.splits != second.splits

    def test_empty_set_raises(self) -> None:
        with pytest.raises(SplitError):
            assign_splits([])

    def test_a_single_group_cannot_be_split_without_leaking(self) -> None:
        records = [source(f"s{i}", speaker_id="spk0") for i in range(4)]

        with pytest.raises(SplitError):
            assign_splits(records)

    def test_min_train_speakers_floor_is_enforced(self) -> None:
        records = [source(f"s{i}", speaker_id=f"spk{i % 3}") for i in range(6)]
        config = SplitConfig(min_train_speakers=4)

        with pytest.raises(SplitError):
            assign_splits(records, config)


class TestHoldouts:
    """Cross-generator (and friends) holdouts are reserved for test, ahead of ratios."""

    def _corpus(self) -> list[SourceRecord]:
        held = [
            source("g1a", speaker_id="spkH1", generator_id="g1"),
            source("g1b", speaker_id="spkH2", generator_id="g1"),
        ]
        main = [source(f"m{i}", speaker_id=f"spkM{i}", generator_id="g2") for i in range(7)]
        return held + main

    def test_generator_holdout_is_reserved_for_test(self) -> None:
        config = SplitConfig(cross_generator_holdout=("g1",))
        assignment = assign_splits(self._corpus(), config)

        held = assignment.holdouts["cross_generator_holdout"]
        assert set(held) == {"g1a", "g1b"}
        assert assignment.split_for("g1a") == "test"
        assert assignment.split_for("g1b") == "test"

    def test_unknown_generator_is_not_held_out(self) -> None:
        records = [*self._corpus(), source("unknown", speaker_id="spkU")]
        config = SplitConfig(cross_generator_holdout=("g1",))
        assignment = assign_splits(records, config)

        assert "unknown" not in assignment.holdouts["cross_generator_holdout"]


class TestChannelAndDeviceDisjointness:
    """A required channel/device axis is a grouping constraint, not a check.

    The claim ``docs/dataset-manifest.md`` asks for is that a channel or handset
    population never appears on both sides of the boundary. Splitting by
    ``(speaker, channel)`` cannot deliver that *and* speaker-disjointness -- it
    just moves the leak -- so the group is widened until every required axis holds
    at once.

    A consequence worth stating in the tests: the number of distinct populations
    becomes the number of indivisible groups, so a channel-disjoint split needs at
    least one population per split, and far more than that to be balanced. A
    corpus with two channel conditions cannot produce three splits at all.
    """

    @staticmethod
    def _corpus(*, channels: int = 8) -> list[SourceRecord]:
        """``channels`` populations of one single-file speaker each.

        One file per speaker keeps the speaker axis from merging populations on its
        own, so each channel population really is one component, and one file per
        group keeps the greedy fill from having to round against a group size.
        """
        return [source(f"c{c}", speaker_id=f"spk{c}", channel=f"ch{c}") for c in range(channels)]

    def test_channel_values_never_span_two_splits(self) -> None:
        records = self._corpus()
        assignment = assign_splits(records, SplitConfig(require_channel_disjoint=True))

        by_channel: dict[str, set[str]] = {}
        for record in records:
            by_channel.setdefault(record.channel, set()).add(
                assignment.split_for(record.sample_id)
            )
        assert all(len(splits) == 1 for splits in by_channel.values())

    def test_a_speaker_spanning_two_channels_is_merged_not_split(self) -> None:
        """The merge is what protects the speaker axis, and it must actually happen."""
        records = [*self._corpus(), source("c1-again", speaker_id="spk1", channel="ch1")]
        assignment = assign_splits(records, SplitConfig(require_channel_disjoint=True))

        assert assignment.split_for("c1") == assignment.split_for("c1-again")
        # ch0 and ch1 are now one group, so each is still on a single side.
        assert assignment.split_for("c0") == assignment.split_for("c1")

    def test_device_works_the_same_way(self) -> None:
        records = [source(f"d{d}", speaker_id=f"spk{d}", device=f"dev{d}") for d in range(8)]
        assignment = assign_splits(records, SplitConfig(require_device_disjoint=True))

        by_device: dict[str, set[str]] = {}
        for record in records:
            by_device.setdefault(record.device, set()).add(
                assignment.split_for(record.sample_id)
            )
        assert all(len(splits) == 1 for splits in by_device.values())

    def test_both_axes_together_still_hold(self) -> None:
        records = [
            source(f"c{c}", speaker_id=f"spk{c}", channel=f"ch{c}", device=f"dev{c}")
            for c in range(8)
        ]
        assignment = assign_splits(
            records,
            SplitConfig(require_channel_disjoint=True, require_device_disjoint=True),
        )

        for axis in ("channel", "device"):
            by_value: dict[str, set[str]] = {}
            for record in records:
                by_value.setdefault(getattr(record, axis), set()).add(
                    assignment.split_for(record.sample_id)
                )
            assert all(len(splits) == 1 for splits in by_value.values()), axis

    def test_a_universal_channel_refuses_instead_of_claiming_disjointness(self) -> None:
        records = [source(f"s{i}", speaker_id=f"spk{i}", channel="wideband") for i in range(6)]

        with pytest.raises(SplitError, match="channel-disjoint"):
            assign_splits(records, SplitConfig(require_channel_disjoint=True))

    def test_too_few_populations_cannot_fill_three_splits(self) -> None:
        """Two channels means two indivisible groups, so dev or test comes out empty."""
        records = [
            source(f"c{c}s{s}", speaker_id=f"spk{c}_{s}", channel=f"ch{c}")
            for c in range(2)
            for s in range(6)
        ]

        with pytest.raises(SplitError, match=r"indivisible|empty"):
            assign_splits(records, SplitConfig(require_channel_disjoint=True))

    def test_an_unpublished_axis_is_reported_not_guessed(self) -> None:
        records = [source(f"s{i}", speaker_id=f"spk{i}") for i in range(8)]

        assignment = assign_splits(records, SplitConfig(require_channel_disjoint=True))

        assert any("no source publishes channel" in note for note in assignment.notes)

    def test_unknown_values_do_not_merge_groups(self) -> None:
        """An unpublished channel is not a private channel, so it must not join."""
        records = [*self._corpus(), source("x1", speaker_id="spkX")]
        assignment = assign_splits(records, SplitConfig(require_channel_disjoint=True))

        assert len(assignment.splits) == len(records)
        # The unknown-channel source is a component of its own, and every published
        # channel is still intact -- which is what lets the gate report the axis as
        # weakened rather than broken.
        by_channel: dict[str, set[str]] = {}
        for record in self._corpus():
            by_channel.setdefault(record.channel, set()).add(
                assignment.split_for(record.sample_id)
            )
        assert all(len(splits) == 1 for splits in by_channel.values())

    def test_the_requirement_is_recorded_in_the_notes(self) -> None:
        assignment = assign_splits(self._corpus(), SplitConfig(require_channel_disjoint=True))

        assert any(
            "channel" in note and "disjoint by construction" in note for note in assignment.notes
        )

    def test_group_labels_name_the_axis_they_were_widened_on(self) -> None:
        assignment = assign_splits(self._corpus(), SplitConfig(require_channel_disjoint=True))

        labels = " ".join(assignment.group_splits)
        assert "channel:ch0" in labels
        assert "channel:ch3" in labels

    def test_unchanged_when_the_requirement_is_off(self) -> None:
        """Off means the previous behaviour exactly, labels included."""
        records = self._corpus()

        without = assign_splits(records, SplitConfig())
        with_flag_off = assign_splits(records, SplitConfig(require_channel_disjoint=False))

        assert without.splits == with_flag_off.splits
        assert without.group_splits == with_flag_off.group_splits
        assert set(without.group_splits) == {record.partition_key for record in records}

    def test_a_holdout_keeps_its_whole_group(self) -> None:
        """A reserved speaker whose other files sit in the main pool stays in test."""
        records = [
            source("h1", speaker_id="spkH", channel="ch0", language="fr"),
            source("m1", speaker_id="spkH", channel="ch0"),
            *(
                source(f"p{index + 1}", speaker_id=f"spk{index}", channel=f"ch{index + 1}")
                for index in range(7)
            ),
        ]
        config = SplitConfig(cross_language_holdout=("fr",), require_channel_disjoint=True)

        assignment = assign_splits(records, config)

        # spkH is reserved, so m1 is test-only rather than quietly training on the
        # very speaker the holdout exists to exclude.
        assert assignment.split_for("m1") == "test"
        assert set(assignment.splits) == {record.sample_id for record in records}
        assert len(assignment.group_splits) == 8
        assert any("contained a cross_language_holdout source" in note for note in assignment.notes)


class TestTemporal:
    """Temporal splits order oldest to train, newest to test, when coverage allows."""

    def test_oldest_sources_carry_train_and_newest_test(self) -> None:
        records = [
            source(f"s{i}", speaker_id=f"spk{i}", recorded_at=f"2024-01-{i + 1:02d}")
            for i in range(8)
        ]
        config = SplitConfig(partition_by_temporal=True)
        assignment = assign_splits(records, config)

        train_days = {
            int(record.recorded_at[9:])
            for record in records
            if assignment.split_for(record.sample_id) == TRAIN
        }
        test_days = {
            int(record.recorded_at[9:])
            for record in records
            if assignment.split_for(record.sample_id) == "test"
        }
        assert max(train_days) < min(test_days)

    def test_insufficient_coverage_falls_back_gracefully(self) -> None:
        records = [
            source(f"s{i}", speaker_id=f"spk{i}", recorded_at=f"2024-01-{i + 1:02d}")
            for i in range(7)
        ]
        records += [source("s7", speaker_id="spk7")]  # no recorded_at
        config = SplitConfig(partition_by_temporal=True)
        assignment = assign_splits(records, config)

        assert len(assignment.splits) == 8
        assert any("fell back" in note for note in assignment.notes)


class TestAccessors:
    """The assignment answers split questions, and refuses to guess."""

    def test_split_for_unknown_source_raises(self) -> None:
        assignment = assign_splits(speakers(12))

        with pytest.raises(SplitError):
            assignment.split_for("never-assigned")

    def test_ids_for_unknown_split_raises(self) -> None:
        assignment = assign_splits(speakers(12))

        with pytest.raises(SplitError):
            assignment.ids_for("holdout")


class TestReport:
    """Split reports are stable, grouped, and persist as JSON."""

    def test_report_roundtrips_through_json(self, tmp_path) -> None:
        records = speakers(12)
        assignment = assign_splits(records)
        report = build_split_report(records, assignment)

        payload = json.loads(json.dumps(report.to_dict()))

        assert payload["counts"] == assignment.counts()
        assert payload["per_label"]["train"]["bona_fide"] == assignment.counts()[TRAIN]

    def test_write_split_report_persists_json(self, tmp_path) -> None:
        records = speakers(12)
        report = build_split_report(records, assign_splits(records))
        path = write_split_report(report, tmp_path / "reports" / "split.json")

        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["counts"] == report.counts()
        assert set(json.loads(path.read_text(encoding="utf-8"))["assignment"]) >= {
            "splits",
            "counts",
        }
