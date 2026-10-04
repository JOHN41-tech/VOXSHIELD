"""Determinism and cache-reuse check.

Runs the build twice against the same corpus without clearing anything, and
compares the two runs. Answers two questions the first smoke run could not:
does a rebuild produce the same rows, and does the preprocessing cache actually
get reused.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scratch_e2e_build import ROOT, generate, harmonic  # noqa: F401
from voxshield.data.build import build_dataset
from voxshield.data.config import load_data_config

RATE = 16000


def config_with_root(root: Path):  # type: ignore[no-untyped-def]
    """Load the real config, retargeted at the smoke root."""
    return load_data_config("configs/data.yaml", overrides={"root": str(root)})


def rows_of(result: object) -> list[dict[str, object]]:
    """Row identities, ignoring the build-specific timestamp."""
    return [
        {
            "sample_id": row.sample_id,
            "split": row.split,
            "audio_path": row.audio_path,
            "start_seconds": row.start_seconds,
            "label_index": row.label_index,
            "speaker_id": row.speaker_id,
        }
        for row in result.rows  # type: ignore[attr-defined]
    ]


def main() -> int:
    """Build twice, then compare rows, splits, and cache hits."""
    target = ROOT / "raw" / "voxshield_local"
    if not target.exists() or not any(target.iterdir()):
        generate()

    config = config_with_root(ROOT)
    config = config.__class__(
        **{
            **{f: getattr(config, f) for f in config.__dataclass_fields__},
            "root": str(ROOT),
            "gates": type(config.gates)(min_test_sources=1),
        }
    )

    first = build_dataset(config)
    first_rows = rows_of(first)
    first_manifest = (ROOT / "manifests" / "train.jsonl").read_text(encoding="utf-8")

    second = build_dataset(config)
    second_rows = rows_of(second)

    same_rows = first_rows == second_rows
    same_build_id = first.build_id == second.build_id
    retrain = (ROOT / "manifests" / "train.jsonl").read_text(encoding="utf-8")

    print(f"build_id stable:        {same_build_id} ({first.build_id})")
    print(f"row identities stable:  {same_rows} ({len(first_rows)} rows)")
    if not same_rows:
        for a, b in zip(first_rows, second_rows, strict=False):
            if a != b:
                print("   first :", a)
                print("   second:", b)
                break
    # cache_hits counts sources, not segments: 30 sources can yield 60 windows.
    accepted = len(first.validation.accepted)
    print(f"first  cache hits: {first.cache_hits} of {accepted} sources")
    print(f"second cache hits: {second.cache_hits} of {accepted} sources")
    print(f"cache fully reused:   {second.cache_hits == accepted}")

    # A manifest that carries a fresh created_at is still the same data; compare
    # the row payloads with that field removed.
    def stable(text: str) -> list[dict[str, object]]:
        return [
            {k: v for k, v in json.loads(line).items() if k != "created_at"}
            for line in text.splitlines()
            if line.strip()
        ]

    print(f"train manifest stable:  {stable(first_manifest) == stable(retrain)}")
    return 0 if same_rows and same_build_id else 1


if __name__ == "__main__":
    sys.exit(main())
