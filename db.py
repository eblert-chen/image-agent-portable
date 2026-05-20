"""SQLite database module for task persistence.

Uses aiosqlite for async access with WAL mode so reads don't block writes.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

if getattr(sys, 'frozen', False):
    _BASE_DIR = Path(sys.executable).parent
else:
    _BASE_DIR = Path(__file__).parent

DB_FILE = str(_BASE_DIR / "image_agent.db")


def _db() -> aiosqlite.Connection:
    """Return a fresh async context manager for the database."""
    return aiosqlite.connect(DB_FILE)


async def _setup(conn: aiosqlite.Connection) -> None:
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")


async def init_db() -> None:
    """Create tables and indexes if they don't exist."""
    async with _db() as db:
        await _setup(db)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                width INTEGER DEFAULT 1024,
                height INTEGER DEFAULT 1024,
                max_retries INTEGER DEFAULT 3,
                pass_threshold REAL DEFAULT 0.80,
                enable_scoring INTEGER DEFAULT 1,
                source TEXT DEFAULT 'agent',
                model TEXT DEFAULT '',
                num_images INTEGER DEFAULT 4,
                status TEXT DEFAULT 'pending',
                created_at TEXT NOT NULL,
                result TEXT,
                error TEXT
            )
        """)
        # Migrate existing DBs that lack the new columns
        for col in ("source", "model"):
            try:
                await db.execute(f"ALTER TABLE tasks ADD COLUMN {col} TEXT DEFAULT ''")
            except Exception:
                pass
        try:
            await db.execute("ALTER TABLE tasks ADD COLUMN num_images INTEGER DEFAULT 4")
        except Exception:
            pass
        await db.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC)")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                enhanced_prompt TEXT,
                model TEXT NOT NULL,
                category TEXT,
                image_base64 TEXT NOT NULL,
                image_format TEXT DEFAULT 'jpeg',
                width INTEGER DEFAULT 1024,
                height INTEGER DEFAULT 1024,
                seed INTEGER,
                best_score REAL,
                clip_score REAL,
                aesthetic_score REAL,
                gpt_vision_score REAL,
                source TEXT DEFAULT 'direct',
                created_at TEXT NOT NULL
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_gen_model ON generations(model)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_gen_category ON generations(category)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_gen_source ON generations(source)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_gen_created ON generations(created_at DESC)")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        await db.commit()
    logger.info("Database initialized at %s", DB_FILE)


async def insert_task(
    task_id: str,
    prompt: str,
    width: int,
    height: int,
    max_retries: int,
    pass_threshold: float,
    enable_scoring: bool,
    status: str,
    created_at: str,
    source: str = "agent",
    model: str = "",
    num_images: int = 4,
) -> None:
    async with _db() as db:
        await _setup(db)
        await db.execute(
            """INSERT INTO tasks (task_id, prompt, width, height, max_retries,
               pass_threshold, enable_scoring, source, model, num_images,
               status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (task_id, prompt, width, height, max_retries, pass_threshold,
             int(enable_scoring), source, model, num_images,
             status, created_at),
        )
        await db.commit()


async def get_task(task_id: str) -> dict | None:
    async with _db() as db:
        await _setup(db)
        async with db.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)) as cursor:
            row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


async def list_tasks() -> list[dict]:
    async with _db() as db:
        await _setup(db)
        async with db.execute("SELECT * FROM tasks ORDER BY created_at DESC") as cursor:
            rows = await cursor.fetchall()
    return [_row_to_dict(r) for r in rows]


async def update_task_status(
    task_id: str,
    status: str,
    result: dict | None = None,
    error: str | None = None,
) -> None:
    async with _db() as db:
        await _setup(db)
        if result is not None:
            await db.execute(
                "UPDATE tasks SET status = ?, result = ? WHERE task_id = ?",
                (status, json.dumps(result, ensure_ascii=False), task_id),
            )
        elif error is not None:
            await db.execute(
                "UPDATE tasks SET status = ?, error = ? WHERE task_id = ?",
                (status, error, task_id),
            )
        else:
            await db.execute(
                "UPDATE tasks SET status = ? WHERE task_id = ?",
                (status, task_id),
            )
        await db.commit()


async def delete_task(task_id: str) -> bool:
    async with _db() as db:
        await _setup(db)
        cursor = await db.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
        await db.commit()
    return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Generations (image + prompt history)
# ---------------------------------------------------------------------------

def _compress_image(data_url: str, max_dim: int = 1024) -> str:
    """Resize and re-encode a data URL as JPEG to save DB space."""
    import base64
    from io import BytesIO
    from PIL import Image

    try:
        header, b64 = data_url.split(",", 1)
    except ValueError:
        return data_url
    raw = base64.b64decode(b64)
    img = Image.open(BytesIO(raw)).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


async def insert_generation(
    prompt: str,
    model: str,
    image_base64: str,
    *,
    enhanced_prompt: str = "",
    category: str | None = None,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    best_score: float | None = None,
    clip_score: float | None = None,
    aesthetic_score: float | None = None,
    gpt_vision_score: float | None = None,
    source: str = "direct",
    created_at: str = "",
) -> int:
    compressed = _compress_image(image_base64)
    async with _db() as db:
        await _setup(db)
        cursor = await db.execute(
            """INSERT INTO generations (prompt, enhanced_prompt, model, category,
               image_base64, image_format, width, height, seed,
               best_score, clip_score, aesthetic_score, gpt_vision_score,
               source, created_at)
               VALUES (?, ?, ?, ?, ?, 'jpeg', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (prompt, enhanced_prompt, model, category,
             compressed, width, height, seed,
             best_score, clip_score, aesthetic_score, gpt_vision_score,
             source, created_at),
        )
        await db.commit()
    return cursor.lastrowid


async def list_generations(
    page: int = 1,
    page_size: int = 50,
    model: str | None = None,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    source: str | None = None,
):
    """Return (items, total_count). Items have image_base64 replaced with a tiny thumbnail."""
    wheres: list[str] = []
    params: list = []
    if model:
        wheres.append("model = ?")
        params.append(model)
    if category:
        wheres.append("category = ?")
        params.append(category)
    if date_from:
        wheres.append("created_at >= ?")
        params.append(date_from)
    if date_to:
        wheres.append("created_at <= ?")
        params.append(date_to + "T23:59:59Z")
    if source:
        wheres.append("source = ?")
        params.append(source)
    where = ("WHERE " + " AND ".join(wheres)) if wheres else ""

    # Count
    async with _db() as db:
        await _setup(db)
        async with db.execute(f"SELECT COUNT(*) FROM generations {where}", params) as cur:
            total = (await cur.fetchone())[0]

        # Page
        offset = (page - 1) * page_size
        async with db.execute(
            f"SELECT * FROM generations {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params + [page_size, offset],
        ) as cur:
            rows = await cur.fetchall()

    items = []
    for r in rows:
        d = _row_to_dict(r)
        items.append(d)
    return items, total


async def get_generation(generation_id: int) -> dict | None:
    async with _db() as db:
        await _setup(db)
        async with db.execute("SELECT * FROM generations WHERE id = ?", (generation_id,)) as cur:
            row = await cur.fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


async def delete_generation(generation_id: int) -> bool:
    async with _db() as db:
        await _setup(db)
        cursor = await db.execute("DELETE FROM generations WHERE id = ?", (generation_id,))
        await db.commit()
    return cursor.rowcount > 0


async def get_generation_filter_counts() -> dict:
    """Return {model: count} and {category: count} for filter dropdowns."""
    async with _db() as db:
        await _setup(db)
        async with db.execute("SELECT model, COUNT(*) as cnt FROM generations GROUP BY model ORDER BY cnt DESC") as cur:
            model_counts = {r[0]: r[1] async for r in cur}
        async with db.execute("SELECT category, COUNT(*) as cnt FROM generations WHERE category IS NOT NULL GROUP BY category ORDER BY cnt DESC") as cur:
            cat_counts = {r[0]: r[1] async for r in cur}
    return {"models": model_counts, "categories": cat_counts}


def _row_to_dict(row: aiosqlite.Row) -> dict:
    d = dict(row)
    d["enable_scoring"] = bool(d.get("enable_scoring", 1))
    if d.get("result") and isinstance(d["result"], str):
        d["result"] = json.loads(d["result"])
    # Normalize scoring fields for generations
    for key in ("best_score", "clip_score", "aesthetic_score", "gpt_vision_score"):
        if key in d and d[key] is not None:
            d[key] = round(float(d[key]), 4)
    return d


# ---------------------------------------------------------------------------
# Settings (API keys, config)
# ---------------------------------------------------------------------------

async def get_settings() -> dict[str, str]:
    """Return all stored settings as a dict."""
    async with _db() as db:
        await _setup(db)
        async with db.execute("SELECT key, value FROM settings") as cur:
            return {r[0]: r[1] async for r in cur}


async def set_settings(data: dict[str, str]) -> None:
    """Upsert settings. Pass empty string value to delete a key."""
    async with _db() as db:
        await _setup(db)
        for key, value in data.items():
            if value:
                await db.execute(
                    "INSERT INTO settings(key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, value),
                )
            else:
                await db.execute("DELETE FROM settings WHERE key = ?", (key,))
        await db.commit()
