# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-19

### Added
- Optional web search fallback for queries that cannot be answered by the uploaded documents.
- Status indicators in the sidebar showing agent readiness and web search state.
- Source badges on assistant messages with list of sources.

### Changed
- Agentic architecture orchestrates tool calls autonomously with priority to documents.
- Settings modal for model, keys, and web search toggle.
- Modular package layout.
- Each package now exposes a clean public API.

## [0.0.1] - 2026-04-12

### Added
- Initial release DocChat.
- Hybrid RAG pipeline combining Multi-Query, HyDE, and Reciprocal Rank Fusion (RRF) using LlamaIndex.
- Content extraction from multiple document formats (PDF, TXT, MD, DOCX).
- User interface built with Streamlit.
- Buttons for file and chat history cleanup.
- Externalized application configuration via `config.yaml`.
- Dockerfile and `compose.yml` for containerized deployment.

[0.1.0]: https://github.com/cyruscsc/docchat/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/cyruscsc/docchat/releases/tag/v0.0.1