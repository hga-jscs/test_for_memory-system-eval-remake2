#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark I/O helpers.

Why this exists:
- Many benchmark data files are UTF-8 / UTF-8 with BOM.
- On Windows (notably conda shells), default text encoding may be GBK.
- Using bare ``open(path)`` can therefore raise UnicodeDecodeError or decode text incorrectly.

This module centralizes robust cross-platform text/JSON loading logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, TextIO

# Encoding order: prefer UTF-8 variants first, then legacy fallbacks.
# GBK/latin-1 are kept as last-resort compatibility paths for heterogeneous files.
DEFAULT_TEXT_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "gbk", "latin-1")


def _normalize_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def safe_open_text(path: str | Path, encodings: Iterable[str] = DEFAULT_TEXT_ENCODINGS) -> TextIO:
    """Open a text file with encoding fallback.

    Raises:
        UnicodeDecodeError: if all candidate encodings fail. Error message includes
            file path and attempted encodings for easier debugging.
    """
    path_obj = _normalize_path(path)
    tried: list[str] = []
    decode_errors: list[UnicodeDecodeError] = []

    for encoding in encodings:
        tried.append(encoding)
        try:
            # Trigger decode early so UnicodeDecodeError surfaces here.
            fh = path_obj.open("r", encoding=encoding)
            fh.read(1)
            fh.seek(0)
            if encoding != "utf-8-sig":
                print(
                    f"[IO-DEBUG] Non-default decoding fallback for {path_obj}: encoding={encoding}",
                    flush=True,
                )
            return fh
        except UnicodeDecodeError as err:
            decode_errors.append(err)
            continue

    last = decode_errors[-1] if decode_errors else UnicodeDecodeError("utf-8", b"", 0, 1, "decode failed")
    raise UnicodeDecodeError(
        last.encoding,
        last.object,
        last.start,
        last.end,
        f"{last.reason}; path={path_obj}; attempted_encodings={tried}",
    )


def load_json_with_fallback(path: str | Path, encodings: Iterable[str] = DEFAULT_TEXT_ENCODINGS) -> Any:
    """Load JSON with robust encoding fallback and explicit path-aware errors."""
    path_obj = _normalize_path(path)
    with safe_open_text(path_obj, encodings=encodings) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as err:
            raise json.JSONDecodeError(
                f"{err.msg}; path={path_obj}",
                err.doc,
                err.pos,
            ) from err


def read_text_with_fallback(path: str | Path, encodings: Iterable[str] = DEFAULT_TEXT_ENCODINGS) -> str:
    """Read text content with robust encoding fallback."""
    with safe_open_text(path, encodings=encodings) as f:
        return f.read()
