#!/usr/bin/env python3
"""Report whether the active Python environment can execute CUDA tensors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="exit with status 2 when no CUDA device can execute a tensor operation",
    )
    return parser.parse_args()


def nvidia_smi_status() -> dict[str, object]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": type(exc).__name__}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def main() -> None:
    args = parse_args()
    device_nodes = ["/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm"]
    report: dict[str, object] = {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn_available": torch.backends.cudnn.is_available(),
        "cudnn_version": torch.backends.cudnn.version(),
        "device_nodes": {node: Path(node).exists() for node in device_nodes},
        "nvidia_smi": nvidia_smi_status(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [],
        "tensor_probe_ok": False,
    }

    if report["cuda_available"]:
        devices = []
        try:
            for index in range(int(report["device_count"])):
                properties = torch.cuda.get_device_properties(index)
                left = torch.ones((16, 16), device=index)
                right = torch.ones((16, 16), device=index)
                output = left @ right
                torch.cuda.synchronize(index)
                devices.append(
                    {
                        "index": index,
                        "name": properties.name,
                        "total_memory_bytes": properties.total_memory,
                        "compute_capability": [properties.major, properties.minor],
                        "probe_sum": output.sum().item(),
                    }
                )
            report["devices"] = devices
            report["tensor_probe_ok"] = len(devices) > 0
        except Exception as exc:  # CUDA failures vary by driver/runtime version.
            report["tensor_probe_error"] = "{}: {}".format(type(exc).__name__, exc)

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_cuda and not report["tensor_probe_ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
