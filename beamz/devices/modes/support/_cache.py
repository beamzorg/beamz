"""Bounded, process-local storage for reusable modal solves."""

from __future__ import annotations

import pickle
from collections import OrderedDict
from collections.abc import Callable, Hashable
from threading import RLock
from typing import Generic, TypeVar

T = TypeVar("T")


class _SolveCache(Generic[T]):
    """Bounded LRU; serialized local results isolate callers' mutable arrays."""

    def __init__(self, *, max_entries: int, max_bytes: int):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[Hashable, bytes] = OrderedDict()
        self._lock = RLock()
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.bytes = self.hits = self.misses = 0

    def get_or_compute(self, key: Hashable, compute: Callable[[], T]) -> T:
        # Concurrent identical misses also share one solve.
        with self._lock:
            payload = self._entries.get(key)
            if payload is not None:
                self._entries.move_to_end(key)
                self.hits += 1
                return pickle.loads(payload)
            self.misses += 1
            value = compute()
            if self.max_entries <= 0 or self.max_bytes <= 0:
                return value
            payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            if len(payload) > self.max_bytes:
                return value
            while self._entries and (
                len(self._entries) >= self.max_entries
                or self.bytes + len(payload) > self.max_bytes
            ):
                _, evicted = self._entries.popitem(last=False)
                self.bytes -= len(evicted)
            self._entries[key] = payload
            self.bytes += len(payload)
            return value
