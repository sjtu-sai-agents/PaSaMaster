"""Select private runtime configuration without editing the public template."""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "config.json"


def get_config_path(path=None):
    """Honor explicit configs, then the environment, local config, and template."""
    requested = Path(path).expanduser() if path is not None else DEFAULT_CONFIG
    if not requested.is_absolute():
        requested = ROOT / requested
    if requested.resolve() != DEFAULT_CONFIG.resolve():
        return requested
    override = os.environ.get("PASAMASTER_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    local = ROOT / "config" / "config.local.json"
    return local if local.is_file() else DEFAULT_CONFIG
