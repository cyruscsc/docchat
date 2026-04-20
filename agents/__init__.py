"""Public API for the ``agents`` package.

Re-exports ``DocChatAgent`` so that callers can do::

    from agents import DocChatAgent

instead of reaching into sub-modules directly.
"""

from agents.docchat import DocChatAgent

__all__ = [
    "DocChatAgent",
]
