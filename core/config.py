"""Centralised configuration loader.

Reads ``config.yaml`` from the project root once at import time and exposes
typed sub-dicts for every section so that no other module needs to touch
``yaml`` or resolve filesystem paths.

Usage::

    from core.config import APP_CONFIG, RAG_CONFIG, WEB_SEARCH_CONFIG
    # or via the package shorthand:
    from core import APP_CONFIG
"""

from __future__ import annotations

import os
from typing import TypedDict

import yaml


# ── TypedDict definitions ──────────────────────────────────────────────────────

class AppConfig(TypedDict):
    llm_options: list[str]
    max_file_upload_limit: int
    max_upload_size_mb: int


class RagConfig(TypedDict):
    chunk_size: int
    chunk_overlap: int
    multi_query_variations: int
    retrieval_top_k: int
    rrf_top_k: int
    rrf_k_parameter: int


class WebSearchConfig(TypedDict):
    max_results: int


# ── Load YAML once at import time ──────────────────────────────────────────────

_config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_config_path, "r") as _f:
    _config = yaml.safe_load(_f)

APP_CONFIG: AppConfig = _config["app"]
RAG_CONFIG: RagConfig = _config["rag"]
WEB_SEARCH_CONFIG: WebSearchConfig = _config["web_search"]
