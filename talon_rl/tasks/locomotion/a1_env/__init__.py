# talon_rl/tasks/locomotion/a1_env/__init__.py
# isaaclab.utils.assets reads the Nucleus asset root into a module-level
# constant (NUCLEUS_ASSET_ROOT_DIR) the first time it's imported, which
# happens transitively the moment `.a1_env`/`.a1_env_cfg` below touch
# isaaclab.sim/isaaclab.assets. On this machine that persistent carb setting
# isn't pre-populated by the Kit app config, so it silently resolves to None
# and any USD asset spawn (including TALON_A1_CFG's) fails with
# FileNotFoundError("None/Isaac/...") unless discovery is forced and the
# setting is set BEFORE that first isaaclab import. Task 1's
# vram_sizing.py needed and verified this identical fix on this exact
# machine (2026-09-14, see task-1-report.md) — this is that same fix,
# applied here so every consumer of this package (not just that throwaway
# script) gets it automatically.
import carb
from isaacsim.storage.native import get_assets_root_path

_asset_root = get_assets_root_path()
if _asset_root is None:
    raise RuntimeError("Could not resolve Isaac Sim assets root")
carb.settings.get_settings().set("/persistent/isaac/asset_root/cloud", _asset_root)

import gymnasium as gym

from .a1_env import IsaacLabTalonEnv
from .a1_env_cfg import IsaacLabTalonEnvCfg

gym.register(
    id="Isaac-Talon-A1-v0",
    entry_point=IsaacLabTalonEnv,
    disable_env_checker=True,  # our step()/reset() return this repo's own
                                # (transition_dict, done_array) shape, not
                                # gym's standard tuple — the checker would
                                # reject that as non-conformant.
    kwargs={"cfg": IsaacLabTalonEnvCfg()},
)
