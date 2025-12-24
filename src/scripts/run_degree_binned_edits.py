"""Run sequential editing experiments for a specific degree bin.

This script runs sequential editing on a subset of triples selected by degree ranking.
All triples are sorted by degree (high to low), then a specific range [bin_start_idx, bin_end_idx)
is selected for editing.

Usage:
    python -m src.scripts.run_degree_binned_edits \
        --model-dir outputs/models/gpt_small_no_alias \
        --kg-dir data/kg/ba_no_alias \
        --output-dir outputs/degree_binned/bin_0 \
        --num-steps 1000 \
        --bin-start-idx 0 \
        --bin-end-idx 1000 \
        --num-retain-triples 1000 \
        --layer 0 \
        --v-num-grad-steps 20 \
        --seed 24 \
        --device cuda
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.sequential_edit.config import SeqEditConfig
from src.sequential_edit.runner import run_sequential_edits
from src.sequential_edit.kg_utils import KG


def create_degree_binned_config(args) -> SeqEditConfig:
    """Create configuration for degree-binned editing.

    Args:
        args: Command line arguments

    Returns:
        SeqEditConfig with degree-binned selection mode
    """
    # Load KG to compute degree range for this bin
    kg_path = Path(args.kg_dir) / "corpus.train.txt"
    kg = KG(str(kg_path))

    # Sort triples by degree (high to low)
    sorted_triples = sorted(kg.triples, key=lambda t: t.degree_s, reverse=True)

    # Get triples in the specified bin range
    bin_triples = sorted_triples[args.bin_start_idx:args.bin_end_idx]

    if len(bin_triples) == 0:
        raise ValueError(
            f"No triples in bin range [{args.bin_start_idx}, {args.bin_end_idx}). "
            f"Total triples: {len(sorted_triples)}"
        )

    # Compute degree range for this bin
    max_degree = max(t.degree_s for t in bin_triples)
    min_degree = min(t.degree_s for t in bin_triples)
    avg_degree = sum(t.degree_s for t in bin_triples) / len(bin_triples)

    print(f"\nBin statistics:")
    print(f"  Range: triples [{args.bin_start_idx}, {args.bin_end_idx})")
    print(f"  Count: {len(bin_triples)}")
    print(f"  Degree range: [{min_degree}, {max_degree}]")
    print(f"  Average degree: {avg_degree:.2f}")
    print()

    # Create config with custom selection mode for this bin
    config = SeqEditConfig(
        model_dir=args.model_dir,
        kg_dir=args.kg_dir,
        output_dir=args.output_dir,
        num_steps=len(bin_triples),  # Use all triples in the bin
        num_retain_triples=args.num_retain_triples,
        eval_mode="sample",
        edit_selection_mode="degree_binned",  # Custom mode
        edit_layer=args.layer,
        max_hop=args.max_hop,
        v_num_grad_steps=args.v_num_grad_steps,
        seed=args.seed,
        device=args.device,
    )

    # Add custom bin metadata to config
    config.bin_start_idx = args.bin_start_idx
    config.bin_end_idx = args.bin_end_idx
    config.bin_degree_min = min_degree
    config.bin_degree_max = max_degree
    config.bin_degree_avg = avg_degree

    return config


def sample_edit_cases_for_bin(kg: KG, bin_start_idx: int, bin_end_idx: int, seed: int):
    """Sample edit cases for a specific degree bin.

    Args:
        kg: Knowledge graph
        bin_start_idx: Start index in degree-sorted triples
        bin_end_idx: End index in degree-sorted triples
        seed: Random seed

    Returns:
        List of edit cases (same format as runner.sample_edit_cases)
    """
    import random

    random.seed(seed)

    # Sort triples by degree (high to low)
    sorted_triples = sorted(kg.triples, key=lambda t: t.degree_s, reverse=True)

    # Get triples in the specified bin range
    bin_triples = sorted_triples[bin_start_idx:bin_end_idx]

    # Get all entities (no alias, abstract entities only)
    entities = sorted(set(kg.entities))

    # Build edit cases
    edit_cases = []
    for triple in bin_triples:
        o_old = triple.o

        # Select a different object
        possible_new_objects = [e for e in entities if e != o_old]
        if not possible_new_objects:
            continue

        o_new = random.choice(possible_new_objects)

        edit_cases.append({
            "s": triple.s,
            "r": triple.r,
            "o_old": o_old,
            "o_new": o_new,
            "degree_s": triple.degree_s,
        })

    return edit_cases


def main():
    """Main entry point for degree-binned editing."""
    parser = argparse.ArgumentParser(
        description="Run sequential editing for a specific degree bin"
    )

    # Required arguments
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--kg-dir",
        type=str,
        required=True,
        help="Path to knowledge graph directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        required=True,
        help="Number of sequential edits (bin size)",
    )
    parser.add_argument(
        "--bin-start-idx",
        type=int,
        required=True,
        help="Start index in degree-sorted triple list",
    )
    parser.add_argument(
        "--bin-end-idx",
        type=int,
        required=True,
        help="End index in degree-sorted triple list (exclusive)",
    )

    # Optional arguments
    parser.add_argument(
        "--num-retain-triples",
        type=int,
        default=1000,
        help="Number of unedited triples for retention evaluation (default: 1000)",
    )
    parser.add_argument(
        "--max-hop",
        type=int,
        default=10,
        help="Maximum hop distance for ripple analysis (default: 10)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Layer to edit (default: 0)",
    )
    parser.add_argument(
        "--v-num-grad-steps",
        type=int,
        default=20,
        help="Number of gradient steps for ROME v optimization (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=24,
        help="Random seed (default: 24)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu, default: cuda)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Degree-Binned Sequential Editing")
    print("=" * 80)
    print(f"Model: {args.model_dir}")
    print(f"KG: {args.kg_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Bin range: [{args.bin_start_idx}, {args.bin_end_idx})")
    print(f"Device: {args.device}")
    print("=" * 80)
    print()

    # Create config
    config = create_degree_binned_config(args)

    # Monkey-patch the sample_edit_cases function to use our bin-specific version
    import src.sequential_edit.runner as runner_module

    original_sample_edit_cases = runner_module.sample_edit_cases

    def patched_sample_edit_cases(kg, num_cases, seed, selection_mode, max_hop):
        """Patched version that uses bin-specific sampling."""
        if selection_mode == "degree_binned":
            return sample_edit_cases_for_bin(
                kg,
                args.bin_start_idx,
                args.bin_end_idx,
                seed
            )
        else:
            return original_sample_edit_cases(
                kg, num_cases, seed, selection_mode, max_hop
            )

    runner_module.sample_edit_cases = patched_sample_edit_cases

    # Run sequential edits
    try:
        run_sequential_edits(config)
        print("\n✓ Degree-binned editing completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during editing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Restore original function
        runner_module.sample_edit_cases = original_sample_edit_cases


if __name__ == "__main__":
    main()
