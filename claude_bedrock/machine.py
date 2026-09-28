"""A small, non-secret snapshot of the box running the session."""

from __future__ import annotations

import platform
import socket
import urllib.request


def _imds(path: str, token: str) -> str | None:
    request = urllib.request.Request(
        f"http://169.254.169.254/latest/meta-data/{path}",
        headers={"X-aws-ec2-metadata-token": token},
    )
    try:
        with urllib.request.urlopen(request, timeout=0.4) as response:
            return response.read().decode().strip() or None
    except Exception:
        return None


def _imds_token() -> str | None:
    request = urllib.request.Request(
        "http://169.254.169.254/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    try:
        with urllib.request.urlopen(request, timeout=0.4) as response:
            return response.read().decode().strip() or None
    except Exception:
        return None


def snapshot(region: str) -> dict:
    token = _imds_token()
    instance_id = _imds("instance-id", token) if token else None
    availability_zone = _imds("placement/availability-zone", token) if token else None
    return {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "region": region,
        "availability_zone": availability_zone or "not on EC2",
        "instance_id": instance_id or "not on EC2",
        "credential": "IAM instance role. No API key in this repo.",
    }
