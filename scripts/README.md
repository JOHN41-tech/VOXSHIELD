## Scratch verification scripts

- scratch_e2e_build.py: end-to-end dataset build smoke check (creates temp structure, runs build)
- scratch_determinism.py: determinism check across two builds
- scratch_cli_smoke.py: CLI smoke check for data commands

Run from repo root with: python scripts/scratch_*.py (or as specified). These are dev-only verification helpers; not part of the test suite.
