from __future__ import annotations

import sys
from pathlib import Path

from sglang_omni.utils import import_external_package

_VOXCPM_ENV_VAR = "SGLANG_OMNI_VOXCPM_CODE_PATH"


def import_voxcpm_package():
    try:
        return import_external_package(
            package_name="voxcpm",
            source_env_var=_VOXCPM_ENV_VAR,
            package_subdir="src",
            install_hint=(
                "Install 'voxcpm' or set SGLANG_OMNI_VOXCPM_CODE_PATH to a "
                "VoxCPM source checkout."
            ),
        )
    except ImportError:
        repo_root = Path(__file__).resolve().parents[3]
        fallback = repo_root / "VoxCPM" / "src"
        if fallback.exists():
            fallback_str = str(fallback)
            if fallback_str not in sys.path:
                sys.path.insert(0, fallback_str)
            import voxcpm

            return voxcpm
        raise
