"""API Key manager — loads stored keys from DB and injects into os.environ.

Called once at startup so all service modules pick up stored keys.
Keys set via environment variables take precedence over stored values.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Mapping: DB key name → environment variable name
KEY_MAP: dict[str, str] = {
    "OPENAI_API_KEY": "OPENAI_API_KEY",
    "LLM_PROVIDER": "LLM_PROVIDER",
    "LLM_MODEL": "LLM_MODEL",
    "LLM_BASE_URL": "LLM_BASE_URL",
    "IMAGE_GEN_API_KEY": "IMAGE_GEN_API_KEY",
    "STABILITY_API_KEY": "STABILITY_API_KEY",
    "ZHIPU_API_KEY": "ZHIPU_API_KEY",
    "RUNNINGHUB_API_KEY": "RUNNINGHUB_API_KEY",
    "DOUBAO_API_KEY": "DOUBAO_API_KEY",
    "MODELSCOPE_API_KEY": "MODELSCOPE_API_KEY",
    "POLLINATIONS_API_KEY": "POLLINATIONS_API_KEY",
    "DASHSCOPE_API_KEY": "DASHSCOPE_API_KEY",
    "HUNYUAN_API_KEY": "HUNYUAN_API_KEY",
    "SPARK_APP_ID": "SPARK_APP_ID",
    "SPARK_API_KEY": "SPARK_API_KEY",
    "SPARK_API_SECRET": "SPARK_API_SECRET",
    "QIANFAN_API_KEY": "QIANFAN_API_KEY",
}


async def load_keys_from_db() -> None:
    """Load stored API keys from DB and merge into os.environ.

    Environment variables already set take precedence — stored keys only
    fill in values that aren't already present in the environment.
    """
    try:
        from db import get_settings
        stored = await get_settings()
        loaded = 0
        for db_key, env_var in KEY_MAP.items():
            if db_key in stored and stored[db_key]:
                if env_var not in os.environ or not os.environ[env_var]:
                    os.environ[env_var] = stored[db_key]
                    loaded += 1
                    logger.debug("Injected %s from stored settings", env_var)
        if loaded:
            logger.info("Loaded %d API key(s) from settings database", loaded)
    except Exception as exc:
        logger.warning("Failed to load keys from DB: %s", exc)
