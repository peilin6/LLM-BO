"""Resource guards for the env-only CPU suite; tests may inject explicit fakes."""

import asyncio
import os
import socket
import subprocess

import pytest
import threadpoolctl


@pytest.fixture(autouse=True)
def forbid_external_resources(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU tests must inject a fake instead of accessing external resources")

    async def forbidden_async(*args, **kwargs):
        forbidden()

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_async)
    monkeypatch.setattr(os, "killpg", forbidden)
    monkeypatch.setattr(
        threadpoolctl, "find_library", lambda name: "libc.so.6" if name == "c" else None
    )
