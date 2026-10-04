"""Unit tests for the corpus adapters and the registry.

The properties tested here are the ones whose failure is invisible in a happy-path
build: a speaker id that lost its namespace, a label that got inferred from a
directory when the corpus never published one, a paired clip whose twin landed on
the other side of a split. Each of those produces a pipeline that runs to
completion and reports a headline number measuring nothing.

The corpora are built on disk in ``tmp_path`` with real, decodable audio, so
header probing and container detection are exercised rather than stubbed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxshield.data.adapters.asvspoof import ASVspoofAdapter
from voxshield.data.adapters.base import DatasetAdapter, container_format_for
from voxshield.data.adapters.real_speech import RealSpeechAdapter
from voxshield.data.adapters.wavefake import WaveFakeAdapter
from voxshield.data.config import (
    LICENSE_POLICY_ALLOW_ALL_EXCEPT_RESTRICTED,
    LICENSE_POLICY_REQUIRE_VERIFIED,
    LICENSE_REQUIRES_VERIFICATION,
    LICENSE_RESTRICTED,
    LICENSE_UNKNOWN,
    LICENSE_VERIFIED,
    DataConfig,
    DatasetEntry,
)
from voxshield.data.errors import AdapterError, DatasetConfigError, RegistryError
from voxshield.data.labels import BONA_FIDE, SPOOF
from voxshield.data.registry import DatasetRegistry, adapter_class, adapter_names
from voxshield.data.schema import UNKNOWN, SourceRecord

SAMPLE_RATE = 16_000


def write_wav(path: Path, seconds: float = 0.25, sample_rate: int = SAMPLE_RATE) -> Path:
    """Write a short real WAV so container headers are genuinely probeable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * sample_rate)) / float(sample_rate)
    signal = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    sf.write(str(path), signal.astype(np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return path


def make_entry(
    dataset_id: str,
    adapter: str,
    task: str = "spoof_detection",
    *,
    path: str = "corpus",
    license_status: str = LICENSE_VERIFIED,
    license_verified_by: str = "fixture",
    license: str = "CC-BY-4.0",
    enabled: bool = True,
    metadata: dict[str, str] | None = None,
) -> DatasetEntry:
    """A registry entry with sensible defaults for tests."""
    return DatasetEntry(
        dataset_id=dataset_id,
        name=dataset_id,
        source="test",
        version="1",
        license=license,
        license_status=license_status,
        license_verified_by=license_verified_by,
        task=task,
        enabled=enabled,
        path=path,
        adapter=adapter,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------- ASVspoof


@pytest.fixture
def asvspoof_with_progress(tmp_path: Path) -> ASVspoofAdapter:
    """A 2021-style corpus: ``progress.txt`` plus flattened ``bona``/``spoof``."""
    root = tmp_path / "raw" / "ASVspoof2021"
    write_wav(root / "LA" / "eval" / "bona" / "LA_E_9332881.wav")
    write_wav(root / "LA" / "eval" / "spoof" / "LA_E_9332882.wav")
    (root / "LA" / "eval" / "progress.txt").write_text(
        "SpeakerID, filename, codec, source, attack, trim, subset\n"
        "1272,LA_E_9332881.wav,-,ASVspoof2019LA,-,-,eval\n"
        "1272,LA_E_9332882.wav,low_mdbmp3,ASVspoof2019LA,A07,-,eval\n",
        encoding="utf-8",
    )
    entry = make_entry("asvspoof2021", "asvspoof", path="raw/ASVspoof2021")
    return ASVspoofAdapter(entry, tmp_path)


@pytest.fixture
def asvspoof_directory_only(tmp_path: Path) -> ASVspoofAdapter:
    """A 2019-style corpus: labels from directories, no metadata file."""
    root = tmp_path / "raw" / "ASVspoof2019"
    write_wav(root / "LA" / "eval" / "bona" / "LA_E_1023179.wav")
    write_wav(root / "LA" / "eval" / "spoof" / "LA_E_1023180.wav")
    entry = make_entry("asvspoof2019", "asvspoof", path="raw/ASVspoof2019")
    return ASVspoofAdapter(entry, tmp_path)


class TestASVspoofProgress:
    def test_label_derives_from_attack_column(
        self, asvspoof_with_progress: ASVspoofAdapter
    ) -> None:
        """ASVspoof has no label column; the attack column encodes it."""
        records = {
            record.sample_id.rsplit(":", 1)[-1]: record
            for record in asvspoof_with_progress.discover()
        }
        bona = next(r for r in records.values() if r.audio_path.endswith("9332881.wav"))
        spoof = next(r for r in records.values() if r.audio_path.endswith("9332882.wav"))
        assert bona.label == BONA_FIDE
        assert spoof.label == SPOOF

    def test_speaker_is_published_and_namespaced(
        self, asvspoof_with_progress: ASVspoofAdapter
    ) -> None:
        records = list(asvspoof_with_progress.discover())
        assert all(r.speaker_id.startswith("asvspoof2021:") for r in records)
        assert {r.speaker_id for r in records} == {"asvspoof2021:1272"}

    def test_attack_preserved_and_null_placeholder_dropped(
        self, asvspoof_with_progress: ASVspoofAdapter
    ) -> None:
        records = {r.audio_path: r for r in asvspoof_with_progress.discover()}
        spoof = next(r for r in records.values() if r.label == SPOOF)
        bona = next(r for r in records.values() if r.label == BONA_FIDE)
        assert spoof.attack_type == "A07"
        # "-" is the corpus's null marker, not an attack identifier.
        assert bona.attack_type == UNKNOWN

    def test_bitrate_grid_becomes_channel_not_codec(
        self, asvspoof_with_progress: ASVspoofAdapter
    ) -> None:
        """A published bitrate condition is not a container, and treating it as
        one would make the cross-codec holdout select bitrates."""
        records = list(asvspoof_with_progress.discover())
        spoof = next(r for r in records if r.label == SPOOF)
        assert spoof.channel == "low_mdbmp3"
        # The container comes from the file, not from the progress file's codec
        # column, which names a transmission condition.
        assert spoof.codec == "WAV"

    def test_corpus_edition_is_provenance_not_a_channel(
        self, asvspoof_with_progress: ASVspoofAdapter
    ) -> None:
        """The source column names the corpus edition. Filed as a channel it
        invents a transmission condition and displaces the real one."""
        records = list(asvspoof_with_progress.discover())
        assert all(r.extra["source"] == "ASVspoof2019LA" for r in records)
        assert all(r.channel != "ASVspoof2019LA" for r in records)

    def test_generator_stays_unknown(self, asvspoof_with_progress: ASVspoofAdapter) -> None:
        """ASVspoof publishes attack ids, not generator systems. Inventing a
        generator here would make a cross-generator holdout select attacks."""
        assert all(r.generator_id == UNKNOWN for r in asvspoof_with_progress.discover())

    def test_container_probed_from_header(self, asvspoof_with_progress: ASVspoofAdapter) -> None:
        records = list(asvspoof_with_progress.discover())
        assert all(r.sample_rate == SAMPLE_RATE for r in records)
        assert all(r.duration_seconds is not None and r.duration_seconds > 0 for r in records)
        assert all(r.codec == "WAV" for r in records)

    def test_malformed_metadata_is_an_error_not_a_fallback(self, tmp_path: Path) -> None:
        """A progress file that cannot be parsed must fail loudly.

        Falling back to directory labels would produce a corpus with no speaker
        metadata that still looks healthy, and the split would quietly stop being
        speaker-disjoint.
        """
        root = tmp_path / "raw" / "broken"
        write_wav(root / "LA" / "eval" / "bona" / "x.wav")
        (root / "progress.txt").write_text("this is not a csv header\n1,2,3\n", encoding="utf-8")
        adapter = ASVspoofAdapter(make_entry("broken", "asvspoof", path="raw/broken"), tmp_path)
        with pytest.raises(AdapterError, match="no recognisable header"):
            list(adapter.discover())


class TestASVspoofDirectoryFallback:
    def test_label_from_directory(self, asvspoof_directory_only: ASVspoofAdapter) -> None:
        records = list(asvspoof_directory_only.discover())
        assert sorted(r.label for r in records) == [BONA_FIDE, SPOOF]

    def test_attack_is_not_guessed_from_the_spoof_directory(
        self, asvspoof_directory_only: ASVspoofAdapter
    ) -> None:
        """Every spoof in a subset can come from a different condition. Guessing
        would manufacture a cross-codec holdout that does not exist."""
        spoof = next(r for r in asvspoof_directory_only.discover() if r.label == SPOOF)
        assert spoof.attack_type == UNKNOWN

    def test_unlabelled_file_is_rejected_by_the_schema(self, tmp_path: Path) -> None:
        root = tmp_path / "raw" / "unlabelled"
        write_wav(root / "stray" / "audio.wav")
        adapter = ASVspoofAdapter(
            make_entry("unlabelled", "asvspoof", path="raw/unlabelled"), tmp_path
        )
        with pytest.raises(Exception, match="label"):
            list(adapter.discover())


# --------------------------------------------------------------- WaveFake


@pytest.fixture
def wavefake_corpus(tmp_path: Path) -> WaveFakeAdapter:
    """WaveFake with a gen/orig pair sharing a filename, plus a second vocoder."""
    root = tmp_path / "raw" / "WaveFake"
    write_wav(root / "train" / "orig" / "LJ001-0001.wav")
    write_wav(root / "train" / "gen" / "melgan" / "LJ001-0001.wav")
    write_wav(root / "train" / "gen" / "HiFi-GAN" / "LJ002-0002.wav")
    # Three spellings of one vocoder, which must not become three generators.
    write_wav(root / "train" / "gen" / "hifigan" / "LJ003-0003.wav")
    write_wav(root / "train" / "gen" / "hi_fi_gan" / "LJ004-0004.wav")
    entry = make_entry("wavefake", "wavefake", path="raw/WaveFake")
    return WaveFakeAdapter(entry, tmp_path)


class TestWaveFake:
    def test_label_and_generator_from_the_tree(self, wavefake_corpus: WaveFakeAdapter) -> None:
        records = {r.audio_path: r for r in wavefake_corpus.discover()}
        assert len(records) == 5
        labels = sorted(r.label for r in records.values())
        assert labels == [BONA_FIDE, SPOOF, SPOOF, SPOOF, SPOOF]
        orig = next(r for r in records.values() if r.label == BONA_FIDE)
        assert orig.generator_id == UNKNOWN, "a genuine clip has no generator to record"

    def test_vocoder_spelling_does_not_fork_a_generator(
        self, wavefake_corpus: WaveFakeAdapter
    ) -> None:
        """ "HiFi-GAN", "hifigan", and "hi_fi_gan" are one vocoder. Left distinct
        they become three generators, and a cross-generator holdout can select one
        spelling while reporting coverage of all of them."""
        spoof_stems = {
            r.audio_path.rsplit("/", 1)[-1].removesuffix(".wav")
            for r in wavefake_corpus.discover()
            if r.label == SPOOF
        }
        assert spoof_stems == {"LJ001-0001", "LJ002-0002", "LJ003-0003", "LJ004-0004"}
        generators = {r.generator_id for r in wavefake_corpus.discover() if r.label == SPOOF}
        assert generators == {"melgan", "hi_fi_gan"}

    def test_paired_twins_share_a_parent(self, wavefake_corpus: WaveFakeAdapter) -> None:
        """The gen/ and orig/ clips for one utterance share a filename, so they
        must share a group or the split can put them on opposite sides."""
        records = list(wavefake_corpus.discover())
        twins = [r for r in records if r.audio_path.endswith("LJ001-0001.wav")]
        assert len(twins) == 2
        assert twins[0].parent_id == twins[1].parent_id
        assert twins[0].label != twins[1].label

    def test_speaker_from_librispeech_filename(self, wavefake_corpus: WaveFakeAdapter) -> None:
        speakers = {r.speaker_id for r in wavefake_corpus.discover()}
        assert speakers == {
            "wavefake:LJ001",
            "wavefake:LJ002",
            "wavefake:LJ003",
            "wavefake:LJ004",
        }

    def test_speaker_extraction_can_be_disabled(self, tmp_path: Path) -> None:
        root = tmp_path / "raw" / "wf_nospeaker"
        write_wav(root / "train" / "orig" / "clip.wav")
        adapter = WaveFakeAdapter(
            make_entry(
                "wf_nospeaker",
                "wavefake",
                path="raw/wf_nospeaker",
                metadata={"speaker_pattern": "none"},
            ),
            tmp_path,
        )
        assert all(r.speaker_id == UNKNOWN for r in adapter.discover())

    def test_corpus_split_is_provenance_only(self, wavefake_corpus: WaveFakeAdapter) -> None:
        assert all(r.source_split == "train" for r in wavefake_corpus.discover())


# ------------------------------------------------------------ real speech


@pytest.fixture
def librispeech_corpus(tmp_path: Path) -> RealSpeechAdapter:
    root = tmp_path / "raw" / "LibriSpeech"
    write_wav(root / "dev-clean" / "1272" / "128104" / "1272-128104-0000.wav")
    write_wav(root / "dev-clean" / "1272" / "128104" / "1272-128104-0001.wav")
    write_wav(root / "dev-clean" / "1462" / "128106" / "1462-128106-0000.wav")
    (root / "dev-clean" / "1272" / "128104" / "1272-128104.trans.txt").write_text(
        "1272-128104-0000 THIS IS A TRANSCRIPT\n", encoding="utf-8"
    )
    entry = make_entry("librispeech", "real_speech", task="real_speech", path="raw/LibriSpeech")
    return RealSpeechAdapter(entry, tmp_path)


class TestRealSpeech:
    def test_layout_autodetected(self, librispeech_corpus: RealSpeechAdapter) -> None:
        assert librispeech_corpus.resolved_layout() == "librispeech"

    def test_speaker_and_chapter_derived_from_the_tree(
        self, librispeech_corpus: RealSpeechAdapter
    ) -> None:
        records = list(librispeech_corpus.discover())
        assert {r.speaker_id for r in records} == {
            "librispeech:1272",
            "librispeech:1462",
        }
        chapters = {r.extra["chapter"] for r in records}
        assert chapters == {"128104", "128106"}

    def test_utterances_of_one_chapter_share_a_parent(
        self, librispeech_corpus: RealSpeechAdapter
    ) -> None:
        """A chapter is one continuous reading session, so it is the unit that
        must not straddle a split -- grouped more tightly than the speaker."""
        records = list(librispeech_corpus.discover())
        same_chapter = [
            r
            for r in records
            if r.audio_path.endswith(("-0000.wav", "-0001.wav")) and r.speaker_id.endswith("1272")
        ]
        assert len(same_chapter) == 2
        assert same_chapter[0].parent_id == same_chapter[1].parent_id

    def test_all_records_are_bona_fide(self, librispeech_corpus: RealSpeechAdapter) -> None:
        assert all(r.label == BONA_FIDE for r in librispeech_corpus.discover())

    def test_language_is_published_as_a_corpus_fact(
        self, librispeech_corpus: RealSpeechAdapter
    ) -> None:
        assert {r.language for r in librispeech_corpus.discover()} == {"en"}

    def test_common_voice_publishes_no_speaker(self, tmp_path: Path) -> None:
        """client_id identifies a contributor, not a speaker, and the validated
        release breaks the mapping. Treating it as a speaker would fake
        speaker-disjointness support."""
        root = tmp_path / "raw" / "cv"
        write_wav(root / "clips" / "en" / "a.mp3", sample_rate=16_000)
        (root / "validated.tsv").write_text(
            "client_id\tpath\tup_votes\tsentence\nabc123\tclips/en/a.mp3\t7\thello\n",
            encoding="utf-8",
        )
        adapter = RealSpeechAdapter(
            make_entry(
                "cv",
                "real_speech",
                task="real_speech",
                path="raw/cv",
                metadata={"layout": "common_voice"},
            ),
            tmp_path,
        )
        records = list(adapter.discover())
        assert len(records) == 1
        assert records[0].speaker_id == UNKNOWN
        assert records[0].language == "en"
        assert records[0].extra["votes"] == "7"

    def test_flat_layout_yields_labelled_records_and_nothing_else(self, tmp_path: Path) -> None:
        root = tmp_path / "raw" / "flat"
        write_wav(root / "one.wav")
        write_wav(root / "two.wav")
        adapter = RealSpeechAdapter(
            make_entry(
                "flat",
                "real_speech",
                task="real_speech",
                path="raw/flat",
                metadata={"layout": "flat"},
            ),
            tmp_path,
        )
        records = list(adapter.discover())
        assert {r.label for r in records} == {BONA_FIDE}
        assert all(r.speaker_id == UNKNOWN for r in records)
        assert all(r.language == UNKNOWN for r in records)

    def test_unknown_layout_is_rejected(self, tmp_path: Path) -> None:
        adapter = RealSpeechAdapter(
            make_entry("bad", "real_speech", task="real_speech", metadata={"layout": "imagenet"}),
            tmp_path,
        )
        with pytest.raises(AdapterError, match="layout"):
            adapter.resolved_layout()


# ------------------------------------------------------- shared behaviour


class TestAdapterBase:
    def test_ids_are_namespaced_by_dataset(self, tmp_path: Path) -> None:
        """Two corpora both containing 1.wav must not collide on a sample id."""
        ids = []
        for name in ("alpha", "beta"):
            root = tmp_path / "raw" / name
            write_wav(root / "train" / "orig" / "1.wav")
            adapter = WaveFakeAdapter(make_entry(name, "wavefake", path=f"raw/{name}"), tmp_path)
            ids.extend(r.sample_id for r in adapter.discover())
        assert len(set(ids)) == 2

    def test_discovery_order_is_deterministic(self, wavefake_corpus: WaveFakeAdapter) -> None:
        first = [r.sample_id for r in wavefake_corpus.discover()]
        second = [r.sample_id for r in wavefake_corpus.discover()]
        assert first == second

    def test_hidden_directories_are_not_data(self, tmp_path: Path) -> None:
        root = tmp_path / "raw" / "wf_hidden"
        write_wav(root / "train" / "orig" / "real.wav")
        write_wav(root / ".cache" / "train" / "orig" / "junk.wav")
        adapter = WaveFakeAdapter(
            make_entry("wf_hidden", "wavefake", path="raw/wf_hidden"), tmp_path
        )
        assert [r.audio_path for r in adapter.discover()] == ["raw/wf_hidden/train/orig/real.wav"]

    def test_max_files_caps_discovery(self, wavefake_corpus: WaveFakeAdapter) -> None:
        assert len(list(wavefake_corpus.discover(max_files=2))) == 2

    def test_missing_corpus_reports_absent_rather_than_raising(self, tmp_path: Path) -> None:
        adapter = WaveFakeAdapter(make_entry("gone", "wavefake", path="raw/nowhere"), tmp_path)
        assert adapter.is_available() is False
        with pytest.raises(Exception, match="not available"):
            adapter.require_available()

    def test_unreadable_header_yields_unknown_not_a_crash(self, tmp_path: Path) -> None:
        """A truncated file must still appear in the inventory, so validation can
        report it. Aborting discovery would suppress the very report needed."""
        root = tmp_path / "raw" / "wf_corrupt"
        path = root / "train" / "orig" / "broken.wav"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not a wav file at all")
        adapter = WaveFakeAdapter(
            make_entry("wf_corrupt", "wavefake", path="raw/wf_corrupt"), tmp_path
        )
        records = list(adapter.discover())
        assert len(records) == 1
        assert records[0].duration_seconds is None
        assert records[0].sample_rate is None

    def test_container_format_is_a_claim_not_a_verification(self) -> None:
        assert container_format_for("x.wav") == "WAV"
        assert container_format_for("x.flac") == "FLAC"
        assert container_format_for("x.mp3") == UNKNOWN
        assert container_format_for("x") == UNKNOWN

    def test_task_mismatch_is_rejected_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(AdapterError, match="task"):
            WaveFakeAdapter(make_entry("wrong", "wavefake", task="real_speech"), tmp_path)

    def test_adapter_name_mismatch_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(AdapterError, match="adapter"):
            WaveFakeAdapter(make_entry("wrong", "asvspoof"), tmp_path)

    def test_describe_reports_a_missing_layout_clearly(self, tmp_path: Path) -> None:
        """Zero rows from a wrong path and zero rows from an empty corpus look
        identical unless the description says which markers were expected."""
        root = tmp_path / "raw" / "mislabelled"
        write_wav(root / "some" / "where" / "a.wav")
        adapter = ASVspoofAdapter(
            make_entry("mislabelled", "asvspoof", path="raw/mislabelled"), tmp_path
        )
        description = adapter.describe()
        assert description.available is True
        assert description.audio_file_count == 1
        assert any("expected layout" in w for w in description.warnings)

    def test_describe_reports_absent_corpus(self, tmp_path: Path) -> None:
        adapter = ASVspoofAdapter(make_entry("absent", "asvspoof", path="raw/absent"), tmp_path)
        description = adapter.describe()
        assert description.available is False
        assert description.audio_file_count == 0
        assert any("does not exist" in w for w in description.warnings)

    def test_describe_marks_a_truncated_count_as_a_lower_bound(self, tmp_path: Path) -> None:
        root = tmp_path / "raw" / "big"
        for index in range(5):
            write_wav(root / "train" / "orig" / f"{index}.wav")
        adapter = WaveFakeAdapter(
            make_entry("big", "wavefake", path="raw/big", metadata={"describe_max_files": "2"}),
            tmp_path,
        )
        description = adapter.describe()
        assert description.audio_file_count == 2
        assert any("lower bound" in w for w in description.warnings)
        assert "lower bound" in description.notes


# ------------------------------------------------------------- registry


class TestRegistry:
    def test_all_adapters_resolve(self) -> None:
        assert adapter_names() == ("asvspoof", "real_speech", "wavefake")
        for name in adapter_names():
            assert issubclass(adapter_class(name), DatasetAdapter)

    def test_unknown_adapter_raises(self) -> None:
        with pytest.raises(RegistryError, match="unknown dataset adapter"):
            adapter_class("imagenet")

    def test_absent_corpus_is_skipped_with_a_reason(self, tmp_path: Path) -> None:
        config = DataConfig(root=str(tmp_path), datasets=(make_entry("gone", "wavefake"),))
        registry = DatasetRegistry(config)
        assert registry.selected() == ()
        reason = registry.skipped()[0]
        assert reason.reason == "not_found"
        assert "gone" in reason.dataset_id

    def test_licence_is_checked_before_presence(self, tmp_path: Path) -> None:
        """A corpus that is both unlicensed and absent is reported as unlicensed.
        The licence blocks the project regardless of machine; the absence does not."""
        config = DataConfig(
            root=str(tmp_path),
            license_policy=LICENSE_POLICY_REQUIRE_VERIFIED,
            datasets=(
                make_entry(
                    "absent_unlicensed",
                    "wavefake",
                    license_status=LICENSE_UNKNOWN,
                    license="unknown",
                ),
            ),
        )
        assert DatasetRegistry(config).skipped()[0].reason == "license"

    def test_require_verified_refuses_an_unverified_corpus(self, tmp_path: Path) -> None:
        root = tmp_path / "raw" / "corpus"
        write_wav(root / "train" / "orig" / "a.wav")
        entry = make_entry(
            "unverified",
            "wavefake",
            path="raw/corpus",
            license_status=LICENSE_REQUIRES_VERIFICATION,
            license="custom",
        )
        strict = DataConfig(
            root=str(tmp_path), license_policy=LICENSE_POLICY_REQUIRE_VERIFIED, datasets=(entry,)
        )
        assert DatasetRegistry(strict).selected() == ()

        permissive = DataConfig(
            root=str(tmp_path),
            license_policy=LICENSE_POLICY_ALLOW_ALL_EXCEPT_RESTRICTED,
            datasets=(entry,),
        )
        assert len(DatasetRegistry(permissive).selected()) == 1

    def test_restricted_is_refused_under_every_policy(self, tmp_path: Path) -> None:
        root = tmp_path / "raw" / "corpus"
        write_wav(root / "train" / "orig" / "a.wav")
        entry = make_entry(
            "blocked",
            "wavefake",
            path="raw/corpus",
            license_status=LICENSE_RESTRICTED,
            license="academic-only",
        )
        for policy in (
            LICENSE_POLICY_REQUIRE_VERIFIED,
            LICENSE_POLICY_ALLOW_ALL_EXCEPT_RESTRICTED,
        ):
            config = DataConfig(root=str(tmp_path), license_policy=policy, datasets=(entry,))
            assert DatasetRegistry(config).selected() == ()

    def test_disabled_entry_is_skipped(self, tmp_path: Path) -> None:
        root = tmp_path / "raw" / "corpus"
        write_wav(root / "train" / "orig" / "a.wav")
        config = DataConfig(
            root=str(tmp_path),
            datasets=(make_entry("off", "wavefake", path="raw/corpus", enabled=False),),
        )
        assert DatasetRegistry(config).skipped()[0].reason == "disabled"

    def test_unknown_adapter_is_a_config_error(self, tmp_path: Path) -> None:
        config = DataConfig(root=str(tmp_path), datasets=(make_entry("x", "imagenet"),))
        reason = DatasetRegistry(config).skipped()[0]
        assert reason.reason == "unknown_adapter"

    def test_duplicate_dataset_id_is_rejected(self, tmp_path: Path) -> None:
        """dataset_id is the speaker namespace and part of the build id, so two
        corpora sharing one produces records that disagree about their origin.

        Caught when the configuration is loaded rather than when the registry is
        built: a duplicate is a typo in the config, and failing at load means it
        cannot be missed by a caller that never constructs a registry.
        """
        with pytest.raises(DatasetConfigError, match="duplicate dataset_id"):
            DataConfig(
                root=str(tmp_path),
                datasets=(
                    make_entry("same", "wavefake", path="raw/a"),
                    make_entry("same", "asvspoof", path="raw/b"),
                ),
            )

    def test_report_names_what_was_skipped(self, tmp_path: Path) -> None:
        """A build that silently used one of three corpora is indistinguishable
        from one where the other two were forgotten."""
        root = tmp_path / "raw" / "here"
        write_wav(root / "train" / "orig" / "a.wav")
        config = DataConfig(
            root=str(tmp_path),
            datasets=(
                make_entry("here", "wavefake", path="raw/here"),
                make_entry("absent", "wavefake", path="raw/gone"),
            ),
        )
        report = DatasetRegistry(config).report()
        assert report["selected"] == ["here"]
        assert [s["dataset_id"] for s in report["skipped"]] == ["absent"]

    def test_discover_all_returns_records_and_descriptions(self, tmp_path: Path) -> None:
        root = tmp_path / "raw" / "here"
        write_wav(root / "train" / "orig" / "a.wav")
        write_wav(root / "train" / "gen" / "melgan" / "a.wav")
        config = DataConfig(
            root=str(tmp_path),
            datasets=(
                make_entry("here", "wavefake", path="raw/here"),
                make_entry("absent", "asvspoof", path="raw/gone"),
            ),
        )
        records, skips, descriptions = DatasetRegistry(config).discover_all()
        assert len(records) == 2
        assert all(isinstance(r, SourceRecord) for r in records)
        assert [s.reason for s in skips] == ["not_found"]
        assert len(descriptions) == 1
        assert descriptions[0].dataset_id == "here"

    def test_max_files_per_dataset_is_applied(self, tmp_path: Path) -> None:
        root = tmp_path / "raw" / "many"
        for index in range(4):
            write_wav(root / "train" / "orig" / f"{index}.wav")
        config = DataConfig(
            root=str(tmp_path),
            max_files_per_dataset=2,
            datasets=(make_entry("many", "wavefake", path="raw/many"),),
        )
        records, _skips, _descriptions = DatasetRegistry(config).discover_all()
        assert len(records) == 2

    def test_task_mismatch_between_entry_and_adapter(self, tmp_path: Path) -> None:
        """A real-speech corpus declared behind a spoof adapter would produce a
        corpus with no negatives."""
        root = tmp_path / "raw" / "here"
        write_wav(root / "train" / "orig" / "a.wav")
        config = DataConfig(
            root=str(tmp_path),
            datasets=(
                DatasetEntry(
                    dataset_id="mismatch",
                    name="mismatch",
                    source="test",
                    version="1",
                    license="CC-BY-4.0",
                    license_status=LICENSE_VERIFIED,
                    license_verified_by="fixture",
                    # real_speech data declared as a spoof_detection entry.
                    task="spoof_detection",
                    enabled=True,
                    path="raw/here",
                    adapter="real_speech",
                ),
            ),
        )
        with pytest.raises(RegistryError, match="task"):
            DatasetRegistry(config).build_adapters()
