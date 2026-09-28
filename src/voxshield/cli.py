"""Command-line entry point.

Serves the API and, for operators, exposes the two things you need during an
incident: whether a model is loaded, and what a specific request decided.

The CLI deliberately has no subcommand that writes audio to disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

from voxshield.monitoring import configure_logging

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
            "no audit store configured; set "
            f"{AUDIT_STORE_ENV} or pass --store",
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

    return parser


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
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
