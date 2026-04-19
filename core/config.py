"""Centralised configuration loader.

Reads ``config.yaml`` from the project root once at import time and exposes
typed sub-dicts for every section so that no other module needs to touch
``yaml`` or resolve filesystem paths.
"""

from __future__ import annotations

import os
import yaml

_config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_config_path, "r") as _f:
    _config = yaml.safe_load(_f)

APP_CONFIG: dict = _config["app"]
RAG_CONFIG: dict = _config["rag"]
WEB_SEARCH_CONFIG: dict = _config["web_search"]
