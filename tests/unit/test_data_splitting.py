"""Unit tests for split assignment.

The property under test is whole-group (speaker-disjoint by construction)
assignment: no known speaker, and no parent recording, may appear on more than
one split side. Secondary properties matter because a split is an artefact an
operator diffes: determinism (same seed, same corpus, same result), stable
reports, and an honest fallback when temporal data cannot support the claim.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime, timedelta

import pytest

from voxshield.data.config import SplitConfig
from voxshield.data.errors import SplitError
from voxshield.data.labels import BONA_FIDE, SPOOF
from voxshield.data.schema import UNKNOWN, SourceRecord
from voxshield.data.splitting import (
    DEV,
    SPLIT_NAMES,
    TEST,
    TRAIN,
    assign_splits,
    build_split_report,
    write_split_report,
)


def source(
    sample_id: str,
    *,
    label: str = BONA_FIDE,
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
        label=label,
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


def classes(bona: int, spoof: int, *, size: int = 1) -> list[SourceRecord]:
    """Two classes as single-class groups of ``size`` sources each.

    One class per group is the case stratification has to get right on its own:
    every class is separately divisible, so a corpus lopsided *between* classes
    cannot be excused by indivisibility.
    """
    records: list[SourceRecord] = []
    for label, count in ((BONA_FIDE, bona), (SPOOF, spoof)):
        for index in range(count):
            for take in range(size):
                records.append(
                    source(
                        f"{label[0]}{index}_{take}", label=label, speaker_id=f"{label[0]}{index}"
                    )
                )
    return records


def mix(records: list[SourceRecord], assignment) -> dict[str, dict[str, int]]:
    """Per-split label counts, for asserting on class composition."""
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLIT_NAMES}
    for record in records:
        counts[assignment.splits[record.sample_id]][record.label] += 1
    return {split: dict(counted) for split, counted in counts.items()}


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


class TestClassStratification:
    """Every split owes each class its share of the corpus, not just its size.

    The failure this prevents is quiet: a capacity fill that hits the size ratios
    exactly while handing ``dev`` a single class, which is a split that cannot
    calibrate, cannot pick a threshold, and cannot report a false-positive rate.
    """

    def test_every_split_receives_both_classes_on_a_lopsided_corpus(self) -> None:
        records = classes(5, 15)
        assignment = assign_splits(records, SplitConfig(min_train_speakers=2), seed=7)
        for split in (DEV, TEST):
            assert mix(records, assignment)[split].keys() == {BONA_FIDE, SPOOF}, (
                f"{split} holds {mix(records, assignment)[split]}, so it cannot be "
                "calibrated or scored for false positives"
            )

    def test_the_class_mix_tracks_the_corpus_rather_than_the_split_order(self) -> None:
        records = classes(10, 30)
        corpus_share = 10 / 40
        config = SplitConfig(train_ratio=0.6, dev_ratio=0.2, min_train_speakers=2)
        assignment = assign_splits(records, config, seed=3)
        for split, counts in mix(records, assignment).items():
            held = sum(counts.values())
            assert counts[BONA_FIDE] / held == pytest.approx(corpus_share, abs=0.05), (
                f"{split} drifted to {counts[BONA_FIDE] / held:.0%} bona fide "
                f"against a corpus share of {corpus_share:.0%}"
            )

    def test_stratification_does_not_cost_split_size(self) -> None:
        """The quota is met by measuring demand in sources, so sizes follow it.

        This is the property that makes the feature free: balancing classes
        usually means choosing a weight against the size ratios, and here there
        is no weight because both constraints are the same one summed over
        classes. If this test fails, the design has grown a tuning knob.
        """
        records = classes(12, 36)
        config = SplitConfig(train_ratio=0.6, dev_ratio=0.2, min_train_speakers=2)
        assignment = assign_splits(records, config, seed=11)
        assert len(assignment.ids_for(TRAIN)) / len(records) == pytest.approx(
            config.train_ratio, abs=0.05
        )

    def test_groups_stay_whole_under_stratification(self) -> None:
        records = classes(6, 18, size=3)
        assignment = assign_splits(records, SplitConfig(min_train_speakers=2), seed=5)
        seen: dict[str, set[str]] = {}
        for record in records:
            seen.setdefault(record.speaker_id or record.sample_id, set()).add(
                assignment.splits[record.sample_id]
            )
        assert all(len(splits) == 1 for splits in seen.values()), "a group was split across sides"

    def test_a_uniform_corpus_is_left_exactly_as_it_was(self) -> None:
        """With nothing to balance, the fill still has to hit the ratios exactly."""
        records = speakers(30)
        assignment = assign_splits(records, SplitConfig(min_train_speakers=2), seed=2)
        assert {split: len(assignment.ids_for(split)) for split in (TRAIN, DEV, TEST)} == {
            TRAIN: 21,
            DEV: 5,
            TEST: 4,
        }
        assert mix(records, assignment)[DEV] == {BONA_FIDE: 5}

    def test_a_split_that_cannot_hold_every_class_says_so(self) -> None:
        """Too few minority groups is arithmetic, so it must be reported.

        Two spoof groups of three sources cannot be spread over three splits at
        the requested ratios -- one of them has to go without. Shipping that
        silently and letting the calibrator divide by zero downstream would be
        worse than saying it plainly.
        """
        records = classes(6, 2, size=3)
        assignment = assign_splits(records, SplitConfig(min_train_speakers=2), seed=13)
        assert mix(records, assignment)[DEV].keys() != {BONA_FIDE, SPOOF}
        assert any("no spoof source" in note and DEV in note for note in assignment.notes), (
            f"the missing class was not reported: {assignment.notes}"
        )

    def test_class_mixed_groups_are_placed_whole_and_counted_once(self) -> None:
        """A group holding both classes belongs to neither, so it is not doubled.

        Counting it under each class would overstate every split, so the reported
        mix would be a worse lie than reporting nothing.
        """
        records: list[SourceRecord] = []
        for index in range(6):
            records.append(source(f"mixed{index}", label=BONA_FIDE, speaker_id=f"m{index}"))
            records.append(source(f"mixed{index}s", label=SPOOF, speaker_id=f"m{index}"))
        assignment = assign_splits(records, SplitConfig(min_train_speakers=2), seed=17)

        counted = mix(records, assignment)
        assert sum(sum(counts.values()) for counts in counted.values()) == len(records), (
            f"a source was lost or counted twice: {counted}"
        )
        for index in range(6):
            halves = {
                assignment.splits[f"mixed{index}"],
                assignment.splits[f"mixed{index}s"],
            }
            assert len(halves) == 1, f"group m{index} was split across {halves}"

    def test_turning_stratification_off_restores_the_capacity_fill(self) -> None:
        """The flag has to be able to reproduce the plain size-only fill on demand."""
        records = classes(5, 15)
        config = SplitConfig(min_train_speakers=2, stratify_by_label=False)
        assignment = assign_splits(records, config, seed=7)
        assert sum(1 for note in assignment.notes if "class mix stratified" in note) == 0
        # The sizes are the only thing it still promises, and it keeps them.
        assert len(assignment.ids_for(TRAIN)) / len(records) == pytest.approx(0.7, abs=0.05)

    def test_temporal_ordering_reports_giving_up_stratification(self) -> None:
        """A chronological boundary and a balanced one are different boundaries."""
        start = datetime(2026, 1, 1, tzinfo=UTC)
        dated = [
            source(
                f"t{index}",
                label=BONA_FIDE if index % 4 else SPOOF,
                speaker_id=f"t{index}",
                recorded_at=(start + timedelta(days=index)).isoformat(),
            )
            for index in range(40)
        ]
        config = SplitConfig(min_train_speakers=2, partition_by_temporal=True)
        assignment = assign_splits(dated, config, seed=19)
        assert any("stratification was not applied" in note for note in assignment.notes), (
            f"the trade was silent: {assignment.notes}"
        )
        assert sum(1 for note in assignment.notes if "class mix stratified" in note) == 0


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
            by_channel.setdefault(record.channel, set()).add(assignment.split_for(record.sample_id))
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
            by_device.setdefault(record.device, set()).add(assignment.split_for(record.sample_id))
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
            by_channel.setdefault(record.channel, set()).add(assignment.split_for(record.sample_id))
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
