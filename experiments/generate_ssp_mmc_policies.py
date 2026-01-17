import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ssp_mmc_fsrs.config import DEFAULT_SEED, DEFAULT_W, POLICY_CONFIGS_PATH  # noqa: E402
from ssp_mmc_fsrs.io import load_policy_configs  # noqa: E402
from experiments.run_experiment_lib import (  # noqa: E402
    ensure_output_dirs,
    generate_ssp_mmc_policies,
    setup_environment,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SSP-MMC policy files and surface plots."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for numpy.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_environment(args.seed)
    ensure_output_dirs()
    policy_configs = load_policy_configs(POLICY_CONFIGS_PATH)
    generate_ssp_mmc_policies(policy_configs, DEFAULT_W)


if __name__ == "__main__":
    main()
