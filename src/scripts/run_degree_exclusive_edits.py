"""Run sequential editing with exclusive (non-overlapping) subjects for degree high/low.

This script ensures that subject entities used in degree_high and degree_low
experiments do not overlap. If NUM_STEPS * 2 > unique_subjects, NUM_STEPS is
automatically adjusted to unique_subjects // 2.

Usage:
    python -m src.scripts.run_degree_exclusive_edits \
        --model-dir outputs/models/gpt_small_no_alias \
        --kg-dir data/kg/ba_no_alias \
        --output-dir outputs/degree_exclusive/degree_high \
        --num-steps 1000 \
        --selection-mode degree_high \
        --num-retain-triples 1000 \
        --layer 0 \
        --v-num-grad-steps 20 \
        --seed 24 \
        --device cuda
"""

import argparse
import sys
import random
from pathlib import Path
from typing import List, Dict, Any, Set

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.sequential_edit.config import SeqEditConfig
from src.sequential_edit.runner import run_sequential_edits
from src.sequential_edit.kg_utils import KG


def get_exclusive_subject_ranges(kg: KG, num_steps: int) -> Dict[str, Any]:
    """Compute exclusive subject ranges for degree high and low.

    Args:
        kg: Knowledge graph
        num_steps: Requested number of steps per mode

    Returns:
        Dictionary containing:
            - adjusted_num_steps: Adjusted number of steps (may be less than requested)
            - unique_subjects_high: Set of unique subjects for degree_high
            - unique_subjects_low: Set of unique subjects for degree_low
            - degree_range_high: (min, max) degree range for high
            - degree_range_low: (min, max) degree range for low
    """
    # Get all unique subjects with their degrees
    subject_degrees = {}
    for triple in kg.triples:
        if triple.s not in subject_degrees:
            subject_degrees[triple.s] = triple.degree_s

    # Sort subjects by degree (high to low)
    sorted_subjects = sorted(
        subject_degrees.items(),
        key=lambda x: x[1],
        reverse=True
    )

    unique_subjects_count = len(sorted_subjects)
    print(f"\nTotal unique subjects: {unique_subjects_count}")

    # Check if we need to adjust num_steps
    if num_steps * 2 > unique_subjects_count:
        adjusted_num_steps = unique_subjects_count // 2
        print(f"⚠ NUM_STEPS * 2 ({num_steps * 2}) > unique subjects ({unique_subjects_count})")
        print(f"⚠ Adjusting NUM_STEPS from {num_steps} to {adjusted_num_steps}")
    else:
        adjusted_num_steps = num_steps
        print(f"✓ NUM_STEPS * 2 ({num_steps * 2}) <= unique subjects ({unique_subjects_count})")
        print(f"✓ Using NUM_STEPS = {adjusted_num_steps}")

    # Select top subjects for degree_high
    high_subjects_list = sorted_subjects[:adjusted_num_steps]
    unique_subjects_high = set(s for s, d in high_subjects_list)
    degree_high_min = min(d for s, d in high_subjects_list)
    degree_high_max = max(d for s, d in high_subjects_list)

    # Select bottom subjects for degree_low
    low_subjects_list = sorted_subjects[-adjusted_num_steps:]
    unique_subjects_low = set(s for s, d in low_subjects_list)
    degree_low_min = min(d for s, d in low_subjects_list)
    degree_low_max = max(d for s, d in low_subjects_list)

    # Verify no overlap
    overlap = unique_subjects_high & unique_subjects_low
    if overlap:
        raise ValueError(f"Subject overlap detected: {overlap}")

    print(f"\nDegree High:")
    print(f"  Subjects: {len(unique_subjects_high)}")
    print(f"  Degree range: [{degree_high_min}, {degree_high_max}]")

    print(f"\nDegree Low:")
    print(f"  Subjects: {len(unique_subjects_low)}")
    print(f"  Degree range: [{degree_low_min}, {degree_low_max}]")

    print(f"\n✓ No subject overlap between degree_high and degree_low")

    return {
        "adjusted_num_steps": adjusted_num_steps,
        "unique_subjects_high": unique_subjects_high,
        "unique_subjects_low": unique_subjects_low,
        "degree_range_high": (degree_high_min, degree_high_max),
        "degree_range_low": (degree_low_min, degree_low_max),
    }


def sample_edit_cases_exclusive(
    kg: KG,
    num_cases: int,
    selection_mode: str,
    unique_subjects: Set[str],
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Sample edit cases from triples with subjects in the given set.

    Args:
        kg: Knowledge graph
        num_cases: Number of edit cases to sample
        selection_mode: "degree_high" or "degree_low"
        unique_subjects: Set of allowed subject entities
        seed: Random seed

    Returns:
        List of edit cases
    """
    random.seed(seed)

    # Filter triples to only include those with subjects in unique_subjects
    allowed_triples = [t for t in kg.triples if t.s in unique_subjects]

    print(f"\n[{selection_mode}] Filtering triples:")
    print(f"  Total KG triples: {len(kg.triples)}")
    print(f"  Allowed subjects: {len(unique_subjects)}")
    print(f"  Allowed triples: {len(allowed_triples)}")

    if len(allowed_triples) < num_cases:
        print(f"⚠ Warning: Only {len(allowed_triples)} allowed triples, but {num_cases} requested")
        print(f"  Using all {len(allowed_triples)} triples")
        sampled_triples = allowed_triples
    else:
        # Sample num_cases triples
        sampled_triples = random.sample(allowed_triples, num_cases)

    # Get all entities (no alias)
    entities = sorted(set(kg.entities))

    # Build edit cases
    edit_cases = []
    for triple in sampled_triples:
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

    print(f"  Generated {len(edit_cases)} edit cases")

    return edit_cases


def main():
    """Main entry point for exclusive degree editing."""
    parser = argparse.ArgumentParser(
        description="Run sequential editing with exclusive subjects for degree high/low"
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
        help="Number of sequential edits (will be adjusted if needed)",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        required=True,
        choices=["degree_high", "degree_low"],
        help="Selection mode: degree_high or degree_low",
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
    print("Exclusive Degree Sequential Editing")
    print("=" * 80)
    print(f"Model: {args.model_dir}")
    print(f"KG: {args.kg_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Selection mode: {args.selection_mode}")
    print(f"Requested NUM_STEPS: {args.num_steps}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Load KG
    print("\nLoading knowledge graph...")
    kg_path = Path(args.kg_dir) / "corpus.train.txt"
    kg = KG(str(kg_path))
    print(f"Loaded KG: {kg}")

    # Compute exclusive subject ranges
    ranges = get_exclusive_subject_ranges(kg, args.num_steps)

    adjusted_num_steps = ranges["adjusted_num_steps"]

    # Select appropriate subject set based on mode
    if args.selection_mode == "degree_high":
        unique_subjects = ranges["unique_subjects_high"]
        degree_range = ranges["degree_range_high"]
    else:  # degree_low
        unique_subjects = ranges["unique_subjects_low"]
        degree_range = ranges["degree_range_low"]

    # Create config
    config = SeqEditConfig(
        model_dir=args.model_dir,
        kg_dir=args.kg_dir,
        output_dir=args.output_dir,
        num_steps=adjusted_num_steps,  # Use adjusted value
        num_retain_triples=args.num_retain_triples,
        eval_mode="sample",
        edit_selection_mode=f"{args.selection_mode}_exclusive",  # Mark as exclusive
        edit_layer=args.layer,
        max_hop=args.max_hop,
        v_num_grad_steps=args.v_num_grad_steps,
        seed=args.seed,
        device=args.device,
    )

    # Add metadata to config
    config.exclusive_mode = True
    config.unique_subjects_count = len(unique_subjects)
    config.degree_range_min = degree_range[0]
    config.degree_range_max = degree_range[1]
    config.requested_num_steps = args.num_steps
    config.adjusted_num_steps = adjusted_num_steps

    # Monkey-patch the sample_edit_cases function to use exclusive subjects
    import src.sequential_edit.runner as runner_module

    original_sample_edit_cases = runner_module.sample_edit_cases

    def patched_sample_edit_cases(kg, num_cases, seed, selection_mode, max_hop):
        """Patched version that uses exclusive subject sampling."""
        if "_exclusive" in selection_mode:
            return sample_edit_cases_exclusive(
                kg,
                num_cases,
                args.selection_mode,
                unique_subjects,
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
        print("\n✓ Exclusive degree editing completed successfully!")
        print(f"\nSubject exclusivity:")
        print(f"  Mode: {args.selection_mode}")
        print(f"  Unique subjects used: {len(unique_subjects)}")
        print(f"  Degree range: [{degree_range[0]}, {degree_range[1]}]")
        print(f"  Subjects guaranteed not to overlap with opposite mode")
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
