from pathlib import Path

import talon_rl.assets as assets


def test_talon_assets_data_dir_points_at_assets_data_directory():
    assert assets.TALON_ASSETS_EXT_DIR == Path("talon_rl/assets").resolve()
    assert assets.TALON_ASSETS_DATA_DIR == assets.TALON_ASSETS_EXT_DIR / "data"
    assert assets.TALON_ASSETS_DATA_DIR.is_dir()


def test_talon_assets_metadata_parses_extension_toml():
    assert assets.TALON_ASSETS_METADATA["package"]["version"] == assets.__version__
    assert isinstance(assets.__version__, str)
