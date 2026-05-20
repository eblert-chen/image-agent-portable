"""Usage tracking module — records daily per-model image generation counts.

Persists to a JSON file on disk with atomic writes. Uses an asyncio lock
to serialize concurrent writes from parallel generation tasks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

if getattr(sys, 'frozen', False):
    _BASE_DIR = Path(sys.executable).parent
else:
    _BASE_DIR = Path(__file__).parent

STATS_FILE = _BASE_DIR / "usage_stats.json"
_lock = asyncio.Lock()


def _load() -> dict:
    try:
        if STATS_FILE.exists():
            return json.loads(STATS_FILE.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to load usage_stats.json, starting fresh")
    return {}


def _save(data: dict) -> None:
    tmp = STATS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATS_FILE)


async def record_usage(model: str, count: int = 1) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    async with _lock:
        data = _load()
        day_entry = data.setdefault(today, {})
        day_entry[model] = day_entry.get(model, 0) + count
        _save(data)


def get_daily_stats() -> dict:
    data = _load()
    daily = []
    total_all = 0
    model_totals: dict[str, int] = {}
    for date_str in sorted(data.keys(), reverse=True):
        models = data[date_str]
        day_total = sum(models.values())
        daily.append({"date": date_str, "models": models, "total": day_total})
        total_all += day_total
        for m, c in models.items():
            model_totals[m] = model_totals.get(m, 0) + c
    top_model = max(model_totals, key=model_totals.get) if model_totals else None
    return {
        "daily": daily,
        "summary": {
            "total_images": total_all,
            "total_days": len(daily),
            "top_model": top_model,
            "model_totals": model_totals,
        },
    }
