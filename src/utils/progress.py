"""Progress bar helpers (tqdm with safe fallback)."""

from __future__ import annotations

from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")


def progress(iterable: Iterable[T], *, enabled: bool, desc: Optional[str] = None) -> Iterator[T]:
    if not enabled:
        return iter(iterable)
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        return iter(iterable)
    return tqdm(iterable, desc=desc)
