"""Knowledge Graph generation module."""

from .generator import (
    Triple,
    KnowledgeGraph,
    generate_ba_kg,
)

__all__ = [
    'Triple',
    'KnowledgeGraph',
    'generate_ba_kg',
]
