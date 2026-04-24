from __future__ import annotations

import importlib


class MissingDependencyError(ImportError):
    """Raised when a required optional dependency is missing."""


def require_package(module_name: str, install_hint: str | None = None):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        hint = f" Install with `{install_hint}`." if install_hint else ""
        raise MissingDependencyError(f"Required dependency `{module_name}` is not installed.{hint}") from exc


def optional_package(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None
