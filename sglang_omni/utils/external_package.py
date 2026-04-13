from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType


def resolve_external_package_root(
    *,
    package_name: str,
    source_env_var: str,
    package_subdir: str | None = None,
) -> Path:
    """Resolve an external package source checkout from an env var."""
    raw_path = os.environ.get(source_env_var)
    if not raw_path:
        raise ImportError(
            f"{package_name!r} is not installed and {source_env_var} is not set."
        )

    root = Path(raw_path).expanduser()
    search_root = root / package_subdir if package_subdir else root
    if not search_root.exists():
        raise ImportError(
            f"{source_env_var} points to {search_root}, but that path does not exist."
        )

    return search_root


def import_external_package(
    *,
    package_name: str,
    source_env_var: str | None = None,
    package_subdir: str | None = None,
    install_hint: str | None = None,
) -> ModuleType:
    """Import a package, optionally falling back to a source checkout path."""
    try:
        return importlib.import_module(package_name)
    except ImportError as first_error:
        if source_env_var is None:
            raise

        try:
            search_root = resolve_external_package_root(
                package_name=package_name,
                source_env_var=source_env_var,
                package_subdir=package_subdir,
            )
        except ImportError as resolve_error:
            hint = install_hint or f"Install {package_name!r} or set {source_env_var}."
            raise ImportError(hint) from resolve_error

        search_root_str = str(search_root)
        if search_root_str not in sys.path:
            sys.path.insert(0, search_root_str)

        try:
            return importlib.import_module(package_name)
        except ImportError as second_error:
            hint = install_hint or (
                f"Install {package_name!r} or point {source_env_var} to a source checkout."
            )
            raise ImportError(hint) from second_error
