"""Command-line entry point.

Serves the API and, for operators, exposes the two things you need during an
incident: whether a model is loaded, and what a specific request decided.

The CLI deliberately has no subcommand that writes audio to disk. ``voxshield
data build`` does write standardised segments, and that is not a contradiction:
it writes the dataset the corpus description already asks for, under
``data/processed``, from audio the operator placed in ``data/raw``. It is not a
request-time path and it takes no upload.

Exit codes are part of the contract, because a script reading stdout cannot tell
"the check passed" from "the check never ran". ``process`` uses ``0``/``1``/``2``/
``3`` because an abstention and a refusal are different claims. The ``data``
commands use ``0``/``1``/``2``:

* ``0`` -- ran to completion, and the answer is yes.
* ``1`` -- ran to completion, and the answer is no: a gate failed, leakage was
  found, no file survived validation, or a loader produced no batch. Findings are
  on stdout and under ``data/reports``.
* ``2`` -- could not run: a bad path, an unreadable configuration, or a manifest
  this build cannot interpret.

Collapsing ``1`` into ``0`` would let a script report a passing leakage check
because the check found nothing to compare.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from voxshield.monitoring import configure_logging

if TYPE_CHECKING:
    from voxshield.data.config import DataConfig
    from voxshield.data.paths import DataPaths

__all__ = ["main"]


def _serve(args: argparse.Namespace) -> int:
    """Run the ASGI server."""
    import uvicorn

    from voxshield.app import create_app

    uvicorn.run(
        create_app(),
        host=args.host,
        port=args.port,
        log_config=None,
    )
    return 0


def _inspect(args: argparse.Namespace) -> int:
    """Print a stored audit record, or state that none exists."""
    from voxshield.app import AUDIT_STORE_ENV

    path = args.store or Path(os.environ.get(AUDIT_STORE_ENV, ""))
    if not str(path):
        print(
            f"no audit store configured; set {AUDIT_STORE_ENV} or pass --store",
            file=sys.stderr,
        )
        return 2

    from voxshield.storage import JsonlAuditStore

    store = JsonlAuditStore(path)
    record = store.get(args.request_id)
    if record is None:
        print(f"no audit record for request_id {args.request_id!r}", file=sys.stderr)
        return 1
    print(json.dumps(record.as_dict(), indent=2, sort_keys=True))
    return 0


def _formats(args: argparse.Namespace) -> int:
    """Print the intake contract."""
    from voxshield.audio.pipeline import allowed_upload_formats
    from voxshield.config import load_audio_config

    cfg = load_audio_config()
    print(
        json.dumps(
            {
                "allowed_containers": allowed_upload_formats(cfg),
                "allowed_subtypes": sorted(cfg.allowed_subtypes),
                "max_upload_bytes": cfg.max_upload_bytes,
                "max_duration_seconds": cfg.max_duration_seconds,
                "max_channels": cfg.max_channels,
                "canonical_sample_rate_hz": cfg.target_sample_rate,
                "min_speech_seconds": cfg.min_speech_seconds,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _process(args: argparse.Namespace) -> int:
    """Analyse one audio file and print a metadata-only report.

    Reads a file, prints a JSON summary, and writes nothing. The exit code
    distinguishes the three outcomes an operator needs to tell apart: the clip
    was analysed, the clip was analysed but is not scorable, or the clip was
    refused and never analysed.
    """
    from voxshield.audio.process import process_audio
    from voxshield.config import load_audio_config
    from voxshield.errors import InsufficientSpeechError, VoxShieldError

    path = Path(args.path)
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 2

    cfg = load_audio_config()
    if args.short_policy:
        cfg = replace(cfg, short_segment_policy=args.short_policy)
    if args.hop is not None:
        cfg = replace(cfg, segment_hop_seconds=args.hop)

    try:
        payload = path.read_bytes()
        result = process_audio(payload, config=cfg, compute_features=not args.no_features)
    except InsufficientSpeechError as exc:
        # An abstention, not a failure. The clip was read, decoded, and measured;
        # it simply does not carry enough speech to window. Exit 3 keeps that
        # distinct from exit 1, which means the file could not be analysed at
        # all. Collapsing the two would teach an operator that "ask for more
        # audio" and "your file is broken" are the same event.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    except VoxShieldError as exc:
        # The error's own message is written to be safe for a client; the audio
        # never appears in it, so printing it here leaks nothing.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    report = result.metadata()
    if args.compact:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))

    # A refusal and a low score are different claims, so they get different
    # exit codes. Collapsing them is how an operator ends up treating an
    # unusable recording as a clean one.
    return 0 if result.scorable else 3


# ---------------------------------------------------------------------------
# Dataset commands
# ---------------------------------------------------------------------------

#: Ran to completion, and the answer is yes.
DATA_OK = 0
#: Ran to completion, and the answer is no.
DATA_FAILED = 1
#: Could not run at all.
DATA_CANNOT_RUN = 2

#: Where the dataset configuration lives by default. Absence is not an error: the
#: built-in defaults plus the environment are a complete configuration, and every
#: report names which one was used so a reader is never guessing.
DEFAULT_DATA_CONFIG = Path("configs/data.yaml")

#: The resolved tree, in the order a person reads it.
_DATA_DIR_FIELDS = (
    "root",
    "raw",
    "interim",
    "processed",
    "manifests",
    "synthetic",
    "cache",
    "reports",
)


def _emit(payload: dict[str, Any], *, compact: bool) -> None:
    """Print one JSON report on stdout.

    Args:
        payload: The report body.
        compact: One line instead of indented JSON.
    """
    if compact:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


def _data_failure(code: int, category: str, exc: Exception) -> int:
    """Report one data failure on stderr and return its exit code.

    Args:
        code: The exit code this class of failure maps to.
        category: Short label naming the layer that failed.
        exc: The exception raised.

    Returns:
        ``code``, unchanged, so callers can ``return _data_failure(...)``.
    """
    print(f"{category}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return code


def _dispatch_data(args: argparse.Namespace) -> int:
    """Run one ``data`` subcommand and turn its failures into exit codes.

    Centralised here, once, because the mapping from failure to exit code is the
    part an operator or a CI script actually depends on, and seven handlers each
    inventing their own is how two of them end up disagreeing.

    The distinction being drawn is "what did you find?" versus "could you look?".
    A gate refusal, leakage, an empty accepted set, and a corpus that could not
    be segmented are findings: the command ran and the answer was no. A missing
    configuration, an unknown path, or a manifest this build cannot interpret
    mean no answer exists at all, and reporting those as ``1`` would let a
    pipeline record "the leakage check found nothing" when in fact it never ran.

    Args:
        args: Parsed arguments, with ``func`` set to the specific handler.

    Returns:
        The handler's exit code, or ``1``/``2`` derived from the failure.
    """
    from voxshield.data.errors import (
        AdapterError,
        AugmentationError,
        DatasetBuildError,
        DatasetConfigError,
        ManifestError,
    )

    try:
        return int(args.func(args))
    except ManifestError as exc:
        return _data_failure(DATA_CANNOT_RUN, "unreadable manifest", exc)
    except DatasetConfigError as exc:
        return _data_failure(DATA_CANNOT_RUN, "configuration", exc)
    except DatasetBuildError as exc:
        return _data_failure(DATA_FAILED, "dataset", exc)
    except (AdapterError, AugmentationError) as exc:
        return _data_failure(DATA_FAILED, "pipeline", exc)


def _data_config(args: argparse.Namespace) -> tuple[DataConfig, DataPaths, str]:
    """Resolve the dataset configuration for a ``data`` subcommand.

    Precedence for the data root is ``--root``, then ``VOXSHIELD_DATA_ROOT``,
    then the configuration document. An explicit flag beating an environment
    variable is why the variable is stripped from the mapping handed to
    :func:`~voxshield.data.config.load_data_config`, which otherwise gives the
    environment the last word.

    Args:
        args: Parsed arguments carrying ``config``, ``root``, and ``compact``.

    Returns:
        The configuration, its resolved paths, and where the configuration came
        from.

    Raises:
        DatasetConfigError: ``--config`` was given and names no file, or the
            configuration itself is invalid. Both mean the command cannot run,
            and both are operator-fixable, so neither is a finding about the
            corpus.
    """
    from voxshield.data.config import load_data_config
    from voxshield.data.errors import DatasetConfigError

    if args.config is not None:
        target = Path(args.config)
        if not target.is_file():
            msg = f"no dataset configuration at {target}"
            raise DatasetConfigError(msg)
        source = str(target)
    elif DEFAULT_DATA_CONFIG.is_file():
        target = DEFAULT_DATA_CONFIG
        source = str(target)
    else:
        target = None
        source = "built-in defaults (no configs/data.yaml in the working directory)"

    environ = dict(os.environ)
    if args.root:
        environ.pop("VOXSHIELD_DATA_ROOT", None)
    config = load_data_config(
        target,
        overrides={"root": args.root} if args.root else None,
        env=environ,
    )
    return config, config.data_paths(), source


def _data_paths_cmd(args: argparse.Namespace) -> int:
    """Print the resolved data layout and this configuration's identity.

    Decodes nothing and hashes no audio, so it is the command to run first when
    a build points somewhere unexpected. ``content_fingerprint`` is printed
    beside ``config_hash`` because the two answer different questions: the first
    is equal across machines holding the same corpus, the second cites this run.
    """
    from voxshield.data.build import jsonable
    from voxshield.data.manifest import SPLIT_MANIFESTS, dataset_build_id

    config, paths, source = _data_config(args)
    _emit(
        jsonable(
            {
                "config_source": source,
                "root": paths.root,
                "directories": {name: getattr(paths, name) for name in _DATA_DIR_FIELDS[1:]},
                "config_hash": config.config_hash(),
                "content_fingerprint": config.content_fingerprint(),
                "dataset_build_id": dataset_build_id(config),
                "random_seed": config.random_seed,
                "license_policy": config.license_policy,
                "max_files_per_dataset": config.max_files_per_dataset,
                "write_segment_audio": config.write_segment_audio,
                "cache_enabled": config.cache.enabled,
                "datasets_enabled": [entry.dataset_id for entry in config.enabled_datasets()],
                "datasets_excluded": [
                    {"dataset_id": entry.dataset_id, "reason": reason}
                    for entry, reason in config.excluded_datasets()
                ],
                "manifests": {
                    name: paths.manifests / filename
                    for name, filename in sorted(SPLIT_MANIFESTS.items())
                },
                "report_paths": {
                    name: getattr(paths, f"{name}_path")()
                    for name in (
                        "inventory",
                        "validation",
                        "split_report",
                        "quality",
                        "leakage",
                        "build_report",
                        "statistics",
                    )
                },
            }
        ),
        compact=args.compact,
    )
    return DATA_OK


def _discover_cmd(args: argparse.Namespace) -> int:
    """Inventory the configured corpora and report duplicates and unreadables.

    Decodes nothing beyond what hashing needs, so this is the cheap way to answer
    "is my corpus even visible to this project" before committing to a build.
    """
    from voxshield.data.build import discover_corpus, jsonable

    config, paths, source = _data_config(args)
    inventory, surviving, registry = discover_corpus(config, paths)

    _emit(
        jsonable(
            {
                "config_source": source,
                "root": paths.root,
                "stats": inventory.stats,
                "registry": registry.report(),
                "kept": len(surviving),
                "kept_ids": sorted(record.sample_id for record in surviving),
                "duplicates": inventory.duplicates,
                "unreadable": inventory.unreadable,
                "report_path": paths.inventory_path(),
            }
        ),
        compact=args.compact,
    )
    return DATA_OK


def _validate_cmd(args: argparse.Namespace) -> int:
    """Decide which discovered files may become training samples.

    Exits ``1`` when nothing survives. That is a finding about the corpus, not a
    crash: a directory of short or silent or wrong-format files is exactly what
    this command exists to say out loud, and exiting ``0`` there would let a build
    pipeline continue toward an empty manifest.
    """
    from voxshield.data.build import discover_corpus, jsonable, validate_corpus

    config, paths, source = _data_config(args)
    inventory, surviving, _registry = discover_corpus(config, paths)
    result = validate_corpus(surviving, config, paths, inventory=inventory)

    _emit(
        jsonable(
            {
                "config_source": source,
                "root": paths.root,
                "discovered": len(surviving),
                "accepted": len(result.accepted),
                "rejected": len(result.rejected),
                "duration_seconds": round(result.stats.duration_seconds, 3),
                "speech_seconds": round(result.stats.speech_seconds, 3),
                "accepted_per_dataset": result.stats.accepted_per_dataset,
                "rejected_per_reason": result.stats.rejected_per_reason,
                "unavailable_scopes": result.stats.unavailable_scopes,
                "rejected_samples": [item.to_dict() for item in result.rejected[: args.limit]],
                "rejected_truncated": len(result.rejected) > args.limit,
                "report_path": paths.validation_path(),
            }
        ),
        compact=args.compact,
    )
    return DATA_OK if result.accepted else DATA_FAILED


def _build_cmd(args: argparse.Namespace) -> int:
    """Run the whole build and report what it produced.

    Every stage report is written before a gate refusal, so this command exits
    ``1`` with the evidence already on disk and the manifest set unpublished.
    """
    from voxshield.data.build import build_dataset, jsonable, summarise
    from voxshield.data.cache import PreprocessingCache

    config, paths, source = _data_config(args)
    result = build_dataset(
        config,
        paths=paths,
        split_seed=args.seed,
        cache=PreprocessingCache(paths, enabled=False) if args.no_cache else None,
        enforce_gates=not args.no_gates,
        overwrite=not args.no_overwrite,
    )

    _emit(
        jsonable(
            {
                "config_source": source,
                "summary": summarise(result).to_dict(),
                "build": result.to_dict(),
                "statistics": result.statistics,
                "gates": result.gate_report,
                "leakage": result.leakage,
                "split": result.assignment,
                "reports": [str(path) for path in result.reports],
            }
        ),
        compact=args.compact,
    )
    return DATA_OK if result.gate_passed else DATA_FAILED


def _inspect_cmd(args: argparse.Namespace) -> int:
    """Print a manifest's header, statistics, and a bounded sample of its rows.

    Reads the manifest and nothing else. It deliberately does not stat the audio
    each row points at, so it stays fast on a 400k-row manifest and remains a
    statement about the manifest rather than about the corpus on disk. Use
    ``test-loader`` for the claim that the audio is actually readable.
    """
    from voxshield.data.build import jsonable
    from voxshield.data.manifest import SPLIT_MANIFESTS, compute_statistics, read_manifest
    from voxshield.data.splitting import SPLIT_NAMES

    _config, paths, source = _data_config(args)
    if args.split not in (*SPLIT_NAMES, "all"):
        print(
            f"split must be one of {', '.join(SPLIT_NAMES)} or 'all', got {args.split!r}",
            file=sys.stderr,
        )
        return DATA_CANNOT_RUN
    if args.manifest is not None and args.split != "all":
        print(
            "--manifest names a file whose header already fixes its split; "
            "pass --split all to read it as written",
            file=sys.stderr,
        )
        return DATA_CANNOT_RUN

    target = Path(args.manifest) if args.manifest else paths.manifests / SPLIT_MANIFESTS[args.split]
    if not target.is_file():
        print(f"no manifest at {target}; run 'voxshield data build' first", file=sys.stderr)
        return DATA_CANNOT_RUN

    manifest = read_manifest(target)

    if args.sample is not None:
        match = next((row for row in manifest.samples if row.sample_id == args.sample), None)
        if match is None:
            # An unknown id is a bad argument, not a finding about the corpus, so
            # it is grouped with "bad split" rather than with a gate refusal.
            print(
                f"no sample {args.sample!r} in {target}; it holds "
                f"{len(manifest.samples)} active row(s)",
                file=sys.stderr,
            )
            return DATA_CANNOT_RUN
        _emit(
            jsonable({"config_source": source, "manifest": str(target), "sample": match}),
            compact=args.compact,
        )
        return DATA_OK

    rows = manifest.samples if args.split == "all" else manifest.by_split(args.split)
    payload: dict[str, Any] = {
        "config_source": source,
        "manifest": str(target),
        "requested_split": args.split,
        "header": manifest.header,
        "statistics": compute_statistics(rows, dataset_build_id=manifest.header.dataset_build_id),
        "retired_rows": len(manifest.retired),
    }
    if args.limit > 0:
        payload["sample_rows"] = rows[: args.limit]
    _emit(jsonable(payload), compact=args.compact)
    return DATA_OK


def _check_leakage_cmd(args: argparse.Namespace) -> int:
    """Re-check every identity axis of a manifest, without rebuilding.

    The build already gates on this; running it separately answers "is the
    manifest on disk still disjoint", which is the question worth asking before
    citing a stored corpus. An axis with nothing known to compare is reported as
    unavailable rather than as clean, and ``--strict`` turns that into a failure
    so a corpus cannot be cited on the strength of an axis nobody could check.
    """
    from voxshield.data.build import jsonable
    from voxshield.data.leakage import LEAKAGE_AXES, check_leakage
    from voxshield.data.manifest import SPLIT_MANIFESTS, read_manifest

    _config, paths, source = _data_config(args)
    target = Path(args.manifest) if args.manifest else paths.manifests / SPLIT_MANIFESTS["all"]
    if not target.is_file():
        print(f"no manifest at {target}; run 'voxshield data build' first", file=sys.stderr)
        return DATA_CANNOT_RUN

    manifest = read_manifest(target)
    report = check_leakage(manifest.samples)
    unavailable = sorted(
        check.axis for check in report.checks if not check.available and not check.leaked
    )
    _emit(
        jsonable(
            {
                "config_source": source,
                "manifest": str(target),
                "rows": len(manifest.samples),
                "has_leakage": report.has_leakage,
                "leaked_axes": list(report.leaked_axes),
                "axes_checked": list(LEAKAGE_AXES),
                "unavailable_axes": unavailable,
                "report": report,
            }
        ),
        compact=args.compact,
    )
    if report.has_leakage:
        return DATA_FAILED
    if args.strict and unavailable:
        print(
            "leakage check passed on the axes it could evaluate, but these axes "
            f"are unevaluable and --strict was set: {', '.join(unavailable)}",
            file=sys.stderr,
        )
        return DATA_FAILED
    return DATA_OK


def _test_loader_cmd(args: argparse.Namespace) -> int:
    """Open one split and iterate a few batches, proving the loader yields.

    Exists because "the manifests were written" and "a trainer can read them" are
    different claims. Reports what was actually observed -- row count, batch
    width, labels seen, whether augmentation applied -- rather than asserting
    success.

    Requires Torch, and its absence exits ``1`` with the extra to install, because
    this is the only command that needs it and the other six must keep working
    without it.
    """
    try:
        from voxshield.data.torch_dataset import build_dataloader, load_split, summarise_batches
    except ImportError as exc:  # pragma: no cover -- depends on the install
        print(f"missing optional dependency: {exc}", file=sys.stderr)
        return DATA_CANNOT_RUN

    from voxshield.data.build import jsonable

    config, paths, source = _data_config(args)
    dataset = load_split(
        args.split,
        paths=paths,
        config=config,
        manifest=args.manifest,
        epoch=args.epoch,
        min_coverage=args.min_coverage,
    )
    loader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=config.random_seed,
    )
    summary = summarise_batches(dataset, loader, max_batches=args.batches)

    payload: dict[str, Any] = {
        "config_source": source,
        "split": args.split,
        "row_count": len(dataset),
        "manifest": str(dataset.manifest.path),
        "label_counts": {str(key): value for key, value in dataset.label_counts().items()},
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        **summary,
    }
    if args.samples > 0:
        payload["sample_items"] = [
            _item_summary(dataset[index]) for index in range(min(args.samples, len(dataset)))
        ]
    _emit(jsonable(payload), compact=args.compact)
    return DATA_OK if summary.get("ok") else DATA_FAILED


def _item_summary(item: dict[str, Any]) -> dict[str, Any]:
    """Shape and metadata of one loader item, with no audio in the output.

    Args:
        item: One item from :class:`VoxShieldDataset`.

    Returns:
        Everything except the waveform values, which are the corpus itself and
        would make the report useless as a diffable summary.
    """
    waveform = item["waveform"]
    return {
        "sample_id": item["sample_id"],
        "waveform_shape": list(waveform.shape),
        "waveform_dtype": str(waveform.dtype),
        "length": int(item["length"]),
        "label": int(item["label"]),
        "coverage": item["coverage"],
        "is_padded": bool(item["is_padded"]),
        "augmented": list(item["augmented"]),
        "augment_seed": item.get("augment_seed"),
    }


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(prog="voxshield", description=__doc__)
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level, e.g. DEBUG, INFO, WARNING.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the API server.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_serve)

    inspect = sub.add_parser("inspect", help="Print a stored audit record.")
    inspect.add_argument("request_id", help="Server-generated request identifier.")
    inspect.add_argument("--store", help="Path to the JSONL audit file.")
    inspect.set_defaults(func=_inspect)

    formats = sub.add_parser("formats", help="Print the intake contract.")
    formats.set_defaults(func=_formats)

    process = sub.add_parser(
        "process",
        help="Analyse one audio file and print a metadata-only report.",
    )
    process.add_argument("path", help="Path to the audio file to analyse.")
    process.add_argument(
        "--hop",
        type=float,
        default=None,
        help="Window advance in seconds (default 50%% overlap).",
    )
    process.add_argument(
        "--short-policy",
        choices=("drop", "pad", "keep"),
        default=None,
        help="What to do with a clip shorter than one window.",
    )
    process.add_argument(
        "--no-features",
        action="store_true",
        help="Skip feature extraction, timing only the earlier stages.",
    )
    process.add_argument(
        "--compact",
        action="store_true",
        help="Print one line of JSON instead of indented JSON.",
    )
    process.set_defaults(func=_process)

    _add_data_parser(sub)
    _add_ml_parser(sub)

    return parser


def _add_data_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ``data`` command group and its seven subcommands.

    A group rather than seven top-level commands, because ``inspect`` is already
    taken by the audit-record reader and because the dataset commands share one
    resolution contract: a configuration, a root, and a JSON report. Sharing it as
    a group means the precedence of ``--root`` over ``VOXSHIELD_DATA_ROOT`` is
    stated once, in :func:`_data_config`, instead of seven times in seven
    argument parsers.

    Args:
        sub: The top-level subparser action to register into.
    """

    def common(parser: argparse.ArgumentParser) -> None:
        """Add the options every ``data`` subcommand accepts."""
        parser.add_argument(
            "--config",
            default=None,
            help=(
                f"Dataset configuration (default: {DEFAULT_DATA_CONFIG} when present, "
                "otherwise built-in defaults)."
            ),
        )
        parser.add_argument(
            "--root",
            default=None,
            help=("Data root directory. Overrides VOXSHIELD_DATA_ROOT and the configured root."),
        )
        parser.add_argument(
            "--compact",
            action="store_true",
            help="Print one line of JSON instead of indented JSON.",
        )

    data = sub.add_parser(
        "data",
        help="Build, inspect, and audit the dataset.",
        description=(
            "Dataset commands. Each prints one JSON report on stdout, sends "
            "diagnostics to stderr, and exits 0 for a yes, 1 for a no, 2 for "
            "could-not-run."
        ),
    )
    data_sub = data.add_subparsers(dest="data_command", required=True)

    paths_cmd = data_sub.add_parser(
        "paths",
        help="Print the resolved data layout and this configuration's identity.",
    )
    common(paths_cmd)
    paths_cmd.set_defaults(func=_data_paths_cmd, dispatch=_dispatch_data)

    discover = data_sub.add_parser(
        "discover",
        help="Inventory the configured corpora; report duplicates and unreadables.",
    )
    common(discover)
    discover.set_defaults(func=_discover_cmd, dispatch=_dispatch_data)

    validate = data_sub.add_parser(
        "validate",
        help="Decide which discovered files may become training samples.",
    )
    common(validate)
    validate.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum rejected rows to list in full (default 50).",
    )
    validate.set_defaults(func=_validate_cmd, dispatch=_dispatch_data)

    build = data_sub.add_parser(
        "build",
        help="Run the full build and publish manifests if the gates pass.",
    )
    common(build)
    build.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Split-assignment seed. Defaults to the configuration's random_seed "
            "for this command, which is the seed the build id cites."
        ),
    )
    build.add_argument(
        "--no-cache",
        action="store_true",
        help="Force a cold build, ignoring the preprocessing cache.",
    )
    build.add_argument(
        "--no-gates",
        action="store_true",
        help=(
            "Publish manifests even when a mandatory gate fails. The reports are "
            "written either way; this only stops the refusal."
        ),
    )
    build.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Refuse to replace existing manifests.",
    )
    build.set_defaults(func=_build_cmd, dispatch=_dispatch_data)

    inspect = data_sub.add_parser(
        "inspect",
        help="Print a manifest's header, statistics, and a bounded sample of rows.",
    )
    common(inspect)
    inspect.add_argument(
        "--split",
        default="all",
        help="Split to read: train, dev, test, or all (default all).",
    )
    inspect.add_argument(
        "--manifest",
        default=None,
        help="Explicit manifest path, overriding --split.",
    )
    inspect.add_argument(
        "--sample",
        default=None,
        help="Print one row by sample_id and exit.",
    )
    inspect.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum rows to print (default 10; 0 prints none).",
    )
    inspect.set_defaults(func=_inspect_cmd, dispatch=_dispatch_data)

    leakage = data_sub.add_parser(
        "check-leakage",
        help="Re-check every identity axis of a manifest, without rebuilding.",
    )
    common(leakage)
    leakage.add_argument(
        "--manifest",
        default=None,
        help="Manifest to check (default: data/manifests/all.jsonl).",
    )
    leakage.add_argument(
        "--strict",
        action="store_true",
        help="Fail when an identity axis has nothing known to compare.",
    )
    leakage.set_defaults(func=_check_leakage_cmd, dispatch=_dispatch_data)

    loader = data_sub.add_parser(
        "test-loader",
        help="Open one split and iterate a few batches. Requires Torch.",
    )
    common(loader)
    loader.add_argument("--split", default="train", help="Split to open (default train).")
    loader.add_argument(
        "--manifest",
        default=None,
        help="Explicit manifest path, overriding --split.",
    )
    loader.add_argument("--batch-size", type=int, default=8, help="Samples per batch.")
    loader.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes. 0 reads in this process, which is reproducible.",
    )
    loader.add_argument("--batches", type=int, default=4, help="Batches to iterate.")
    loader.add_argument("--epoch", type=int, default=0, help="Epoch index for augmentation seeds.")
    loader.add_argument(
        "--min-coverage",
        type=float,
        default=0.0,
        help="Drop rows whose speech coverage is below this fraction.",
    )
    loader.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Number of individual items to describe (default 0).",
    )
    loader.set_defaults(func=_test_loader_cmd, dispatch=_dispatch_data)


# ---------------------------------------------------------------------------
# Training commands
# ---------------------------------------------------------------------------

#: A run produced a measured report. The numbers exist.
ML_MEASURED = 0
#: A run started and could not finish. Something is wrong with the code or data.
ML_FAILED = 1
#: Nothing ran, and nothing could have. No corpus, a dry run, or an unrequested
#: test evaluation. Distinct from ``ML_FAILED`` so a pipeline does not record a
#: broken pipeline when the honest answer is an absent dataset.
ML_CANNOT_RUN = 2


def _dispatch_ml(args: argparse.Namespace) -> int:
    """Run one ``ml`` subcommand and map its outcome to an exit code.

    The exit code carries the distinction the whole phase turns on: ``0`` means a
    number was produced, ``2`` means none was and none could have been. A CI
    script that treats ``1`` and ``2`` alike will eventually report a corpus
    problem as a code defect.

    Args:
        args: Parsed arguments, with ``func`` set to the specific handler.

    Returns:
        The handler's exit code.
    """
    try:
        return int(args.func(args))
    except Exception as exc:
        category = "training"
        if args.command == "ml":
            category = f"ml {getattr(args, 'ml_command', '?')}"
        return _data_failure(ML_FAILED, category, exc)


def _ml_train_cmd(args: argparse.Namespace) -> int:
    """Train one baseline and print its run record.

    Reports NOT RUN rather than failing when there is nothing to train on, which
    is this repository's actual state. Both an absent manifest and a manifest
    with no rows produce the same well-formed record and the same exit code, so a
    pipeline can tell "no corpus" from "broken code" without reading stderr.
    """
    from voxshield.data.build import jsonable
    from voxshield.data.errors import VoxShieldError
    from voxshield.data.manifest import read_manifest
    from voxshield.training.config import load_training_config
    from voxshield.training.runner import RunOutcome, not_run_result, run_baseline

    config = load_training_config(args.config)

    try:
        manifest = read_manifest(args.manifest)
    except VoxShieldError as exc:
        # A missing or uninterpretable manifest is "could not run", not "failed".
        result = not_run_result(
            f"no usable manifest at {args.manifest} ({exc}). "
            "This project ships no training corpus.",
            config=config,
        )
        _emit(jsonable(result.to_dict()), compact=args.compact)
        return ML_CANNOT_RUN

    result = run_baseline(
        config,
        manifest,
        root=args.root or config.output_dir,
        dry_run=args.dry_run,
        time_inference=args.time,
        evaluate_test=not args.no_test,
        max_items=args.max_items,
    )

    payload = jsonable(result.to_dict())
    _emit(payload, compact=args.compact)

    if args.record_dir:
        path = result.write(args.record_dir)
        print(f"run record: {path}", file=sys.stderr)

    if result.outcome is RunOutcome.MEASURED:
        return ML_MEASURED
    if result.outcome is RunOutcome.FAILED:
        return ML_FAILED
    return ML_CANNOT_RUN


def _ml_evaluate_cmd(args: argparse.Namespace) -> int:
    """Score a split with a saved artefact, without refitting."""
    from voxshield.data.build import jsonable
    from voxshield.data.manifest import read_manifest
    from voxshield.training.runner import RunOutcome, run_from_artifact

    manifest = read_manifest(args.manifest)
    result = run_from_artifact(
        args.artifact,
        manifest,
        root=args.root,
        split=args.split,
        time_inference=args.time,
    )
    _emit(jsonable(result.to_dict()), compact=args.compact)

    if result.outcome is RunOutcome.MEASURED:
        return ML_MEASURED
    if result.outcome is RunOutcome.FAILED:
        return ML_FAILED
    return ML_CANNOT_RUN


def _ml_models_cmd(args: argparse.Namespace) -> int:
    """List what has actually been trained, or resolve one model in detail.

    Existence is reported separately from usability, so an empty root is a
    measured fact about this machine rather than an error. That distinction is
    the whole reason the registry exists: "no models here" and "the model you
    asked for is not here" need different responses from whoever is looking.
    """
    from voxshield.data.build import jsonable
    from voxshield.training.registry import ModelRegistry, ModelRegistryError

    registry = ModelRegistry.discover(args.root)

    if args.resolve:
        try:
            entry = registry.get(args.resolve)
        except ModelRegistryError as exc:
            _emit(
                jsonable(
                    {
                        "status": "not_found",
                        "model_id": args.resolve,
                        "reason": str(exc),
                        "available": [item.model_id for item in registry],
                    }
                ),
                compact=args.compact,
            )
            return ML_CANNOT_RUN
        _emit(jsonable({"status": "found", **entry.to_dict()}), compact=args.compact)
        return ML_MEASURED

    report = registry.report()
    report["status"] = "measured"
    if not registry:
        # An empty registry is measured, not failed: nothing is wrong, there is
        # simply nothing here. Exit 0 so a pipeline can proceed.
        report["reason"] = "no trained baselines found under this root"
    _emit(jsonable(report), compact=args.compact)
    return ML_MEASURED


def _ml_config_cmd(args: argparse.Namespace) -> int:
    """Validate a recipe and print its identity without touching any audio.

    Cheap enough to run in a pre-commit hook, which is the point: a recipe that
    does not parse should fail before a training run is attempted.
    """
    from voxshield.data.build import jsonable
    from voxshield.training.config import load_training_config

    config = load_training_config(args.config)
    _emit(
        jsonable(
            {
                "config_path": str(args.config),
                "model_id": config.model_id,
                "family": config.model.family,
                "config_hash": config.config_hash(),
                "content_fingerprint": config.content_fingerprint(),
                "features": config.features.to_dict(),
                "model": config.model.to_dict(),
                "train": config.train.to_dict(),
                "threshold": config.threshold.to_dict(),
                "splits": {
                    "train": config.train_split,
                    "dev": config.dev_split,
                    "test": config.test_split,
                },
                "output_dir": str(config.output_dir),
                "notes": list(config.notes),
            }
        ),
        compact=args.compact,
    )
    return ML_MEASURED


def _add_ml_parser(sub: Any) -> None:
    """Register the ``ml`` command group.

    Args:
        sub: The top-level subparser action.
    """
    ml = sub.add_parser(
        "ml",
        help="Train and evaluate the Phase 3 baselines.",
        description=(
            "Training commands. Each prints one JSON record on stdout and exits "
            "0 for a measured result, 1 for a failed run, and 2 for a run that "
            "could not happen. This repository ships no corpus, so an untrained "
            "checkout exits 2 and says NOT RUN rather than inventing a number."
        ),
    )
    ml_sub = ml.add_subparsers(dest="ml_command", required=True)

    train = ml_sub.add_parser(
        "train",
        help="Fit one baseline and evaluate it under the protocol.",
        description=(
            "Fits on train, selects on dev, then scores test exactly once at the "
            "threshold dev chose."
        ),
    )
    train.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ml/mfcc_logreg.yaml"),
        help="Training recipe (default configs/ml/mfcc_logreg.yaml).",
    )
    train.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/all.jsonl"),
        help="Manifest holding the train, dev, and test splits.",
    )
    train.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Data root for relative audio paths. Defaults to output_dir.",
    )
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and featurise the splits, then stop. Fits nothing.",
    )
    train.add_argument(
        "--time",
        action="store_true",
        help="Record per-call latency. Roughly triples run cost.",
    )
    train.add_argument(
        "--no-test",
        action="store_true",
        help="Stop after the dev report, leaving the test split unread.",
    )
    train.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Cap rows per split, for smoke runs.",
    )
    train.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help="Directory for the run record, in addition to stdout.",
    )
    train.add_argument(
        "--compact",
        action="store_true",
        help="Print one line of JSON instead of indented JSON.",
    )
    train.set_defaults(func=_ml_train_cmd, dispatch=_dispatch_ml)

    evaluate = ml_sub.add_parser(
        "evaluate",
        help="Score a split using a saved artefact.",
    )
    evaluate.add_argument("--artifact", type=Path, required=True, help="Artefact directory.")
    evaluate.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/all.jsonl"), help="Manifest path."
    )
    evaluate.add_argument(
        "--root", type=Path, default=Path("."), help="Data root for relative audio paths."
    )
    evaluate.add_argument("--split", default="test", help="Split to score (default test).")
    evaluate.add_argument("--time", action="store_true", help="Record per-call latency.")
    evaluate.add_argument(
        "--compact", action="store_true", help="Print one line of JSON instead of indented JSON."
    )
    evaluate.set_defaults(func=_ml_evaluate_cmd, dispatch=_dispatch_ml)

    config_cmd = ml_sub.add_parser(
        "config",
        help="Validate a recipe and print its identity. Decodes no audio.",
    )
    config_cmd.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ml/mfcc_logreg.yaml"),
        help="Training recipe to validate.",
    )
    config_cmd.add_argument(
        "--compact", action="store_true", help="Print one line of JSON instead of indented JSON."
    )
    config_cmd.set_defaults(func=_ml_config_cmd, dispatch=_dispatch_ml)

    models_cmd = ml_sub.add_parser(
        "models",
        help="List trained baselines found under an artefact root.",
    )
    models_cmd.add_argument(
        "--root",
        type=Path,
        default=Path("data/models"),
        help="Directory whose subdirectories are artefacts.",
    )
    models_cmd.add_argument(
        "--resolve",
        help="Print one model in detail, including its calibration and threshold.",
    )
    models_cmd.add_argument(
        "--compact", action="store_true", help="Print one line of JSON instead of indented JSON."
    )
    models_cmd.set_defaults(func=_ml_models_cmd, dispatch=_dispatch_ml)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    # ``data`` subcommands register a dispatch wrapper alongside their handler so
    # that one function, not seven, decides how a failure becomes an exit code.
    dispatch = getattr(args, "dispatch", None)
    result: int = dispatch(args) if dispatch is not None else int(args.func(args))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
