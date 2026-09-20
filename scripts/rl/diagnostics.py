#!/usr/bin/env python3
"""Single entry point for every eval-only/no-training diagnostic script in
this repo. 2026-09-20: scripts/rl/ had grown to 33 files (train_prelim.py,
sim2sim.py, play.py, plus 29 one-off diagnostic scripts accumulated across
Experiments 2A-2E), each with its own AppLauncher/argparse boilerplate.
Each diagnostic's actual logic is untouched -- moved verbatim into
scripts/rl/_diagnostics_impl/ (git mv where tracked, plain mv for scripts
written this same session) -- this file is a thin dispatcher only, so none
of the tested experiment behavior changed in the move.

Usage:
    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/diagnostics.py <experiment> [experiment args...]
    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/diagnostics.py --list

Example (2D.2's calf-target sweep):
    python scripts/rl/diagnostics.py static-standing --num_envs 64 --steps 120 \\
        --settle_window 20 --kp 25 --calf_target -1.3 --sim_dt 0.01

Each <experiment>'s own --help works normally (forwarded to that script's
own argparse), e.g.:
    python scripts/rl/diagnostics.py static-standing --help
"""

from __future__ import annotations

import importlib
import sys

# subcommand (kebab-case) -> module name under scripts.rl._diagnostics_impl.
# Kept as one flat registry (not auto-discovered) so `--list` output is
# deliberate and stable, not whatever happens to be on disk.
_REGISTRY = {
    "advantage-decomposition": "advantage_decomposition",
    "aggregate-decomposition": "aggregate_decomposition",
    "attractor-phase-metrics": "attractor_phase_metrics",
    "balance-decomposition": "balance_decomposition",
    "balance-progress-trace": "balance_progress_trace",
    "combine-checkpoint-traces": "combine_checkpoint_traces",
    "combine-hip-intervention-traces": "combine_hip_intervention_traces",
    "command-gait-comparison": "command_gait_comparison",
    "command-sensitivity-probe": "command_sensitivity_probe",
    "command-switch-trace": "command_switch_trace",
    "fall-cycle-analysis": "fall_cycle_analysis",
    "gait-activity-ratio": "gait_activity_ratio",
    "gait-joint-trace": "gait_joint_trace",
    "gravity-actuator-consistency": "gravity_actuator_consistency",
    "gravity-finite-diff-check": "gravity_finite_diff_check",
    "ground-contact-timestep-check": "ground_contact_timestep_check",
    "hip-asymmetry-analysis": "hip_asymmetry_analysis",
    "hip-functional-correlation": "hip_functional_correlation",
    "hip-intervention-trace": "hip_intervention_trace",
    "hip-symmetry-intervention": "hip_symmetry_intervention",
    "initial-drop-metrics": "initial_drop_metrics",
    "instantaneous-equilibrium-check": "instantaneous_equilibrium_check",
    "leg-collapse-trace": "leg_collapse_trace",
    "objective-segment-audit": "objective_segment_audit",
    "open-loop-step-response": "open_loop_step_response",
    "prefall-window-analysis": "prefall_window_analysis",
    "progress-leak-counterfactual": "progress_leak_counterfactual",
    "static-standing": "static_standing_diagnostic",
    "torque-authority-ablation": "torque_authority_ablation",
}


def _print_list() -> None:
    print("Available diagnostics (scripts/rl/diagnostics.py <name> [args...]):")
    for name in sorted(_REGISTRY):
        print(f"  {name}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") or sys.argv[1] == "--list":
        _print_list()
        if len(sys.argv) < 2 or sys.argv[1] == "--list":
            return
        return

    experiment = sys.argv[1]
    module_name = _REGISTRY.get(experiment)
    if module_name is None:
        print(f"Unknown diagnostic: {experiment!r}\n")
        _print_list()
        raise SystemExit(1)

    # Forward remaining args to the target script's own argparse, as if it
    # had been invoked directly (sys.argv[0] convention preserved for --help).
    sys.argv = [f"diagnostics.py {experiment}"] + sys.argv[2:]
    module = importlib.import_module(f"scripts.rl._diagnostics_impl.{module_name}")
    module.main()


if __name__ == "__main__":
    main()
