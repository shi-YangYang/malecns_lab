"""Locate assets in this source workspace independently of the caller's cwd."""

from pathlib import Path


def source_project_root() -> Path:
    """Require the repository's src layout; never guess a site-packages root.

    Device experiments use configs, data and logs outside the Python package.
    Run them from this checkout with an editable installation.
    """
    package = Path(__file__).resolve().parent
    for candidate in package.parents:
        if ((candidate / "pyproject.toml").is_file()
                and (candidate / "src" / "cnsbridge").resolve() == package):
            return candidate
    raise RuntimeError("CNSBridge experiments require the source checkout and an editable installation")
