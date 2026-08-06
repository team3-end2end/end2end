"""Compatibility entrypoint for ``python -m reporting.generate``."""

from .__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
