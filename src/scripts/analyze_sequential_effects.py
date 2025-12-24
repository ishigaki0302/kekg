"""Script to analyze and visualize sequential editing results.

This script loads the output of sequential editing experiments and
generates visualizations including time series plots, hop/degree
heatmaps, and failure histograms.

Usage:
    python -m src.scripts.analyze_sequential_effects [--output-dir DIR]
"""

import argparse
from src.sequential_edit.analysis import run_all_plots


def main():
    """Main entry point for analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze sequential editing results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sequential",
        help="Directory containing sequential editing outputs (default: outputs/sequential)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Sequential Editing Analysis")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    print()

    # Run all visualizations
    run_all_plots(args.output_dir)

    print()
    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
