"""Configuration loading.

Loads config/default.yaml, then merges config/local.yaml on top of it if
present (local.yaml is gitignored and holds personal/live overrides).
Also loads .env for secrets via python-dotenv. Nothing in this module
should ever read a secret value and put it in a config dict that later
gets logged -- secrets stay in environment variables and are pulled
directly by the code that needs them (see data/kalshi_client.py).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "default.yaml"
_LOCAL_CONFIG_PATH = _REPO_ROOT / "config" / "local.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    default_path: Path = _DEFAULT_CONFIG_PATH,
    local_path: Path = _LOCAL_CONFIG_PATH,
    load_env: bool = True,
) -> dict[str, Any]:
    """Load and merge configuration.

    Raises FileNotFoundError if the default config is missing -- there is
    no silent fallback to hardcoded defaults, since risk limits living only
    in code would defeat the point of having them in a reviewable config file.
    """
    if load_env:
        load_dotenv(_REPO_ROOT / ".env", override=False)

    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found at {default_path}")

    with open(default_path) as f:
        config = yaml.safe_load(f) or {}

    if local_path.exists():
        with open(local_path) as f:
            local_overrides = yaml.safe_load(f) or {}
        config = _deep_merge(config, local_overrides)

    return config


def is_live_trading_authorized(config: dict[str, Any]) -> bool:
    """Two independent gates must both agree before live orders are ever sent.

    1. config.execution.mode == "live"  (a deliberate, reviewed config change)
    2. env var KALSHI_LIVE_TRADING_CONFIRMED == "true" (a runtime confirmation
       that isn't persisted in version control, so a config file alone can
       never flip the system into live trading)
    """
    config_says_live = config.get("execution", {}).get("mode") == "live"
    env_confirms = os.environ.get("KALSHI_LIVE_TRADING_CONFIRMED", "false").lower() == "true"
    return config_says_live and env_confirms
