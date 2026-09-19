"""Run one Phase 3 mission in demo or OpenAI live mode."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.crew import DEFAULT_MISSION, run_demo_crew, run_live_crew


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission", default=DEFAULT_MISSION)
    parser.add_argument("--live", action="store_true", help="Call OpenAI instead of local demo mode")
    parser.add_argument("--model", help="Override OPENAI_MODEL")
    args = parser.parse_args()

    try:
        result = (
            run_live_crew(args.mission, args.model)
            if args.live
            else run_demo_crew(args.mission)
        )
    except RuntimeError as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Mode: {result.mode}")
    print(f"Model: {result.model_name}")
    print("\nFinal recommendation:\n")
    print(result.final_output)
    if result.usage:
        print(f"\nUsage: {result.usage}")


if __name__ == "__main__":
    main()
