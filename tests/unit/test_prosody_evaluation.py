import numpy as np

from voxshield.evaluation.prosody.evaluate_features import (
    evaluate_arrays,
    evaluate_dataset_directory,
    manifest_for_paths,
)


def test_empty_dataset_is_not_available(tmp_path) -> None:
    report = evaluate_dataset_directory(tmp_path)
    assert report.status == "NOT AVAILABLE"
    assert report.sample_count == 0
    assert report.mean_latency_ms is None


def test_fixture_evaluation_reports_metrics() -> None:
    t = np.arange(20_000, dtype=np.float32) / 16_000
    report = evaluate_arrays([0.2 * np.sin(2 * np.pi * 130 * t).astype(np.float32)])
    assert report.status == "COMPLETED"
    assert report.sample_count == 1
    assert report.mean_latency_ms is not None


def test_manifest_is_sorted_and_hashed(tmp_path) -> None:
    first = tmp_path / "b.wav"
    second = tmp_path / "a.wav"
    first.write_bytes(b"b")
    second.write_bytes(b"a")
    manifest = manifest_for_paths([first, second])
    assert [record["path"] for record in manifest] == [str(second), str(first)]
    assert len(manifest[0]["sha256"]) == 64
