from __future__ import annotations

from pathlib import Path


def load_workspace_config(config_path: str | None = None) -> dict:
    """Load the shared workspace YAML configuration."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load workspace_config.yaml. Install with: pip install pyyaml"
        ) from exc

    default_path = Path(__file__).with_name("configs/workspace_config.yaml")
    path = Path(config_path) if config_path else default_path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration root must be a mapping (YAML dictionary).")

    return config


def get_section(config: dict, section: str) -> dict:
    value = config.get(section)
    if not isinstance(value, dict):
        raise ValueError(
            f"Missing or invalid section '{section}' in workspace configuration.")
    return value
