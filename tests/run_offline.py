#!/usr/bin/env python3
"""Run synthetic tests with every real socket/DNS connection disabled."""
from pathlib import Path
import contextlib
import io
import socket
import sys
import unittest


def main():
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    blocked = []

    def deny(*args, **kwargs):
        blocked.append("network_attempt")
        raise AssertionError("Network access is forbidden in the offline test suite")

    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.socket.sendto = deny
    socket.create_connection = deny
    socket.getaddrinfo = deny
    suite = unittest.defaultTestLoader.discover(str(project / "tests"), pattern="test_*.py")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        print(output.getvalue())
    print(f"Offline verification: {result.testsRun} tests; failures={len(result.failures)}, "
          f"errors={len(result.errors)}; blocked network attempts={len(blocked)}.")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
