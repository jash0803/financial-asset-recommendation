"""Two-tier caching: Streamlit in-memory + joblib disk persistence."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import joblib

from config import CacheConfig


class CacheManager:
    """Manages disk-level caching with TTL-based invalidation.

    Streamlit-level caching (``@st.cache_data`` / ``@st.cache_resource``)
    is applied via decorators in the UI layer; this class handles the
    heavier joblib persistence that survives app restarts.
    """

    def __init__(self, cfg: CacheConfig | None = None):
        self._cfg = cfg or CacheConfig()
        if self._cfg.enabled:
            self._cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        safe = hashlib.md5(key.encode()).hexdigest()
        return self._cfg.cache_dir / f"{safe}.joblib"

    def get(self, key: str) -> Any | None:
        if not self._cfg.enabled:
            return None
        path = self._key_path(key)
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self._cfg.ttl_seconds:
            path.unlink(missing_ok=True)
            return None
        return joblib.load(path)

    def set(self, key: str, value: Any) -> None:
        if not self._cfg.enabled:
            return
        joblib.dump(value, self._key_path(key))

    def invalidate(self, key: str) -> None:
        self._key_path(key).unlink(missing_ok=True)

    def clear(self) -> None:
        if self._cfg.cache_dir.exists():
            for f in self._cfg.cache_dir.glob("*.joblib"):
                f.unlink()
