#!/usr/bin/env python3
"""Offline cache transfer; run only after stopping the original process."""
from slr_pipeline.resume import main

if __name__ == "__main__":
    raise SystemExit(main())
