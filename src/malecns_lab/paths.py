"""Locate assets in this source workspace independently of the caller's cwd."""

from pathlib import Path


def project_asset_path(path: str | Path) -> Path:
    """Resolve model and calibration assets against the source checkout."""
    path = Path(path).expanduser()
    return (path if path.is_absolute() else source_project_root() / path).resolve()


def source_project_root() -> Path:
    """Require the repository's src layout; never guess a site-packages root.

    Device experiments use configs, data and logs outside the Python package.
    Run them from this checkout with an editable installation.
    """
    package = Path(__file__).resolve().parent
    for candidate in package.parents:
        if ((candidate / "pyproject.toml").is_file()
                and (candidate / "src" / "malecns_lab").resolve() == package):
            return candidate
    raise RuntimeError("MaleCNS Lab experiments require the source checkout and an editable installation")
