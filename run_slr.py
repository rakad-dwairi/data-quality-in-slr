#!/usr/bin/env python3
"""Single entry point. No stages run until this file is invoked as a program."""
from slr_pipeline.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
