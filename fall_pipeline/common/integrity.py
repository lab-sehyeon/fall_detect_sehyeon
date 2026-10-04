"""Immutable-input checks used by the recovered experiment protocols."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hashes(expected: Mapping[str, str], root: str | Path = ".") -> dict:
    root = Path(root)
    actual = {}
    mismatches = []
    for name, wanted in expected.items():
        path = root / name
        got = sha256_file(path) if path.is_file() else None
        actual[name] = got
        if got != wanted:
            mismatches.append({"path": name, "expected": wanted, "actual": got})
    return {"passed": not mismatches, "actual": actual, "mismatches": mismatches}


def hash_named_tensors(named_tensors: Iterable[tuple[str, object]]) -> str:
    """Hash tensor names, shapes, dtypes and bytes without changing tensors."""
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()
