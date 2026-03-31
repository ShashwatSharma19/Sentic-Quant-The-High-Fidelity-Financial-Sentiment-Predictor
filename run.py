"""
run.py
------
The unified pipeline runner for the Algorithmic Trading project.
Executes data ingestion, feature engineering, model training, and backtesting in sequence.
"""

import logging
from src.config import settings

# Since these modules are currently scripted at the module-level instead of via classes,
# we would trigger them by executing their __main__ functions, or by refactoring them eventually.
# For now, we will import them or run them as subprocesses if they aren't fully modular yet.

import subprocess
import sys
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s")
logger = logging.getLogger("PipelineRunner")

def main():
    parser = argparse.ArgumentParser(description="Algorithmic Trading Pipeline")
    parser.add_argument("--step", choices=["ingest", "features", "train", "train_rl", "backtest", "all"], default="all")
    args = parser.parse_args()

    # Determine which steps to run
    steps_to_run = []
    if args.step == "all":
        steps_to_run = ["src.data_ingestion", "src.feature_engineering", "src.model", "src.rl_agent", "src.backtest"]
    elif args.step == "ingest":
        steps_to_run = ["src.data_ingestion"]
    elif args.step == "features":
        steps_to_run = ["src.feature_engineering"]
    elif args.step == "train":
        steps_to_run = ["src.model"]
    elif args.step == "train_rl":
        steps_to_run = ["src.rl_agent"]
    elif args.step == "backtest":
        steps_to_run = ["src.backtest"]

    for module in steps_to_run:
        logger.info(f"=== Running Phase: {module} ===")
        # Run module as python -m
        result = subprocess.run([sys.executable, "-m", module], check=False)
        if result.returncode != 0:
            logger.error(f"Pipeline stalled. {module} exited with code {result.returncode}.")
            sys.exit(result.returncode)

    logger.info("=== Entire Pipeline Finished Successfully! ===")


if __name__ == "__main__":
    main()
