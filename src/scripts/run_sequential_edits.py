"""Script to run sequential editing experiments.

This script executes sequential knowledge editing using ROME and logs
the resulting metrics, ripple effects, and accuracy data.

Usage:
    python -m src.scripts.run_sequential_edits [options]
"""

import argparse
from src.sequential_edit.config import SeqEditConfig
from src.sequential_edit.runner import run_sequential_edits


def main():
    """Main entry point for sequential editing."""
    parser = argparse.ArgumentParser(
        description="Sequential Knowledge Editing with ROME"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="outputs/models/gpt_mini_ba",
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--kg-dir",
        type=str,
        default="data/kg/ba",
        help="Path to knowledge graph directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sequential",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of sequential editing steps",
    )
    parser.add_argument(
        "--num-eval-triples",
        type=int,
        default=1000,
        help=(
            "Number of triples for evaluation "
            "(used only when --eval-mode=sample)"
        ),
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="sample",
        choices=["sample", "all", "all-excl-edits"],
        help=(
            "How to select evaluation triples: "
            "'sample' = random sample, "
            "'all' = all triples (including edit cases), "
            "'all-excl-edits' = all triples excluding edit cases"
        ),
    )
    parser.add_argument(
        "--max-hop",
        type=int,
        default=10,
        help="Maximum hop distance for analysis (default: 10, usually no need to change for small KGs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for computation",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer to edit (if None, auto-locate for each edit)",
    )
    parser.add_argument(
        "--v-num-grad-steps",
        type=int,
        default=5,
        help="Number of gradient steps for ROME right vector optimization (default: 5)",
    )
    parser.add_argument(
        "--edit-selection-mode",
        type=str,
        default="random",
        choices=["random", "degree_high", "degree_low", "hop_high", "hop_low"],
        help=(
            "How to select edit cases: "
            "'random' = random sampling, "
            "'degree_high' = subjects with high degree, "
            "'degree_low' = subjects with low degree, "
            "'hop_high' = subjects with high average hop distance, "
            "'hop_low' = subjects with low average hop distance"
        ),
    )
    parser.add_argument(
        "--num-retain-triples",
        type=int,
        default=1000,
        help="Number of unedited triples for retention evaluation (default: 1000)",
    )

    args = parser.parse_args()

    # Create configuration
    config = SeqEditConfig(
        num_steps=args.num_steps,
        num_eval_triples=args.num_eval_triples,
        num_retain_triples=args.num_retain_triples,
        max_hop=args.max_hop,
        edit_method="rome",
        edit_selection_mode=args.edit_selection_mode,
        edit_layer=args.layer,
        v_num_grad_steps=args.v_num_grad_steps,
        model_dir=args.model_dir,
        kg_dir=args.kg_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        eval_mode=args.eval_mode,
    )

    print("=" * 80)
    print("Sequential Editing Experiment")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Steps: {config.num_steps}")
    print(f"  Edit selection mode: {config.edit_selection_mode}")
    print(f"  Eval mode: {config.eval_mode}")
    print(f"  Eval triples (if sample): {config.num_eval_triples}")
    print(f"  Retain triples (unedited): {config.num_retain_triples}")
    print(f"  Max hop: {config.max_hop}")
    print(f"  Edit method: {config.edit_method}")
    print(
        f"  Edit layer: {config.edit_layer if config.edit_layer is not None else 'auto-locate'}"
    )
    print(f"  ROME v_num_grad_steps: {config.v_num_grad_steps}")
    print(f"  Model dir: {config.model_dir}")
    print(f"  KG dir: {config.kg_dir}")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Seed: {config.seed}")
    print(f"  Device: {config.device}")
    print("=" * 80)
    print()

    # Run sequential editing
    run_sequential_edits(config)

    print()
    print("=" * 80)
    print("Experiment complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()