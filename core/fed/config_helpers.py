"""Config helpers: YAML loading, shuffle_seed lookup, dataset name extraction.

Module-level functions (no state) — called by FederatedServer during init
and by ScriptBuilder when building per-client scripts.
"""

import logging
import os
import re
from typing import Any, Dict, Optional

import yaml

_logger = logging.getLogger(__name__)


def load_config(config_path: str, logger=None) -> Dict[str, Any]:
    """Load a YAML config file. Raises on failure."""
    log = logger or _logger
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        log.info(f"Successfully loaded config from {config_path}")
        return config
    except Exception as e:
        log.error(f"Failed to load config from {config_path}: {str(e)}")
        raise


def get_shuffle_seed(config: Dict[str, Any], logger=None) -> Optional[int]:
    """Resolve shuffle_seed with precedence: env var → federated.data_sharding → data."""
    log = logger or _logger
    try:
        env = os.environ.get('SHUFFLE_SEED')
        if env:
            return int(env)
        fed_ds = config.get('federated', {}).get('data_sharding', {})
        if 'shuffle_seed' in fed_ds:
            return fed_ds['shuffle_seed']
        data = config.get('data', {})
        if 'shuffle_seed' in data:
            return data['shuffle_seed']
        return None
    except Exception as e:
        log.warning(f"Error reading shuffle_seed from config: {e}")
        return None


def extract_dataset_name(config_path: str, logger=None) -> str:
    """Extract a dataset name like 'verl-agent_webshop_grpo' from a config filename.

    Example: fed_webshop_grpo_total-100_...yaml -> verl-agent_webshop_grpo
    Fallback (single underscore after fed_): fed_alfworld_... -> verl-agent_alfworld
    """
    log = logger or _logger
    name = os.path.basename(config_path).replace('.yaml', '')

    m = re.match(r'fed_([^_]+_[^_]+)_', name)
    if m:
        return f"verl-agent_{m.group(1)}"

    m = re.match(r'fed_([^_]+)_', name)
    if m:
        return f"verl-agent_{m.group(1)}"

    log.warning(f"Could not extract dataset name from config path: {config_path}")
    return "verl-agent"
