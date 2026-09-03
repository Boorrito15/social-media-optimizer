"""Database module for SMO."""

from .database import (
    delete_prediction,
    get_history,
    get_prediction,
    init_db,
    save_prediction,
)

__all__ = [
    "delete_prediction",
    "get_history",
    "get_prediction",
    "init_db",
    "save_prediction",
]
