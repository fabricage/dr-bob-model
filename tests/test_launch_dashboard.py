"""Launcher helpers: do not start Streamlit in unit tests."""

from __future__ import annotations

import socket

from launch_dashboard import port_is_open


def test_port_is_open_false_when_nothing_listens() -> None:
    assert port_is_open("127.0.0.1", 1) is False


def test_port_is_open_true_for_bound_socket() -> None:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        assert port_is_open("127.0.0.1", port) is True
    finally:
        server.close()
