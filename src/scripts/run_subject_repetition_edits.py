"""Run sequential editing with controlled subject repetition.

This script controls how many times each subject appears in the edit cases:
- repetition_count=1: Each subject is selected at most once (no repetition)
- repetition_count=2: Each subject is selected exactly 2 times
- repetition_count=4: Each subject is selected exactly 4 times

Key features:
- Randomly select subjects without replacement
- For each subject, randomly select repetition_count triples
- Each selected triple gets a unique o_new (no collision)
- Total edit cases = num_steps (number of subjects = num_steps / repetition_count)

Usage:
    python -m src.scripts.run_subject_repetition_edits \
        --model-dir outputs/models/gpt_small \
        --kg-dir data/kg/ba \
        --output-dir outputs/subject_rep/no_rep \
        --num-steps 30 \
        --repetition-count 1 \
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
from typing import List, Dict, Any, Set, Tuple
from collections import Counter, defaultdict
from math import ceil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.sequential_edit.config import SeqEditConfig
from src.sequential_edit.runner import run_sequential_edits
from src.sequential_edit.kg_utils import KG


def compute_structures_from_kg(kg: KG) -> Tuple[Dict[str, List], Dict[Tuple[str, str], Set[str]], Set[str]]:
    """Compute auxiliary structures from KG for edit case sampling.

    Args:
        kg: Knowledge graph

    Returns:
        Tuple of:
            - triples_by_s: Dict mapping subject to list of triples
            - sr2os: Dict mapping (s,r) to set of existing objects
            - entities: Set of all entities (subjects + objects)
    """
    triples_by_s = defaultdict(list)
    sr2os = defaultdict(set)
    subjects = set()
    objects = set()

    for t in kg.triples:
        triples_by_s[t.s].append(t)
        sr2os[(t.s, t.r)].add(t.o)
        subjects.add(t.s)
        objects.add(t.o)

    entities = subjects | objects

    return dict(triples_by_s), dict(sr2os), entities


def sample_edit_cases_with_repetition(
    kg: KG,
    num_steps: int,
    repetition_count: int,
    triples_by_s: Dict[str, List],
    sr2os: Dict[Tuple[str, str], Set[str]],
    entities: Set[str],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sample edit cases with controlled subject repetition.

    Args:
        kg: Knowledge graph
        num_steps: Total number of edit cases to generate
        repetition_count: Number of times each subject should be selected
        triples_by_s: Subject to triples mapping
        sr2os: (s,r) to existing objects mapping
        entities: All entities set
        seed: Random seed

    Returns:
        List of edit cases with fields:
            - s, r, o_old, o_new, repetition_count, seed
    """
    rng = random.Random(seed)

    # Calculate number of subjects needed
    # Allow flexible repetition count for the last subject if num_steps is not divisible
    num_subjects = ceil(num_steps / repetition_count)

    print(f"\nSampling edit cases with subject repetition:")
    print(f"  Total edit cases (num_steps): {num_steps}")
    print(f"  Repetition count per subject: {repetition_count}")
    print(f"  Number of subjects needed: {num_subjects}")

    if num_steps % repetition_count != 0:
        last_subject_reps = num_steps % repetition_count
        print(f"  ⚠ Note: Last subject will have {last_subject_reps} repetition(s) (num_steps not divisible by repetition_count)")

    # Get all subjects that have at least repetition_count triples
    valid_subjects = [
        s for s, triples in triples_by_s.items()
        if len(triples) >= repetition_count
    ]

    print(f"  Available subjects with >= {repetition_count} triples: {len(valid_subjects)}")

    if len(valid_subjects) < num_subjects:
        raise ValueError(
            f"Not enough subjects with >= {repetition_count} triples. "
            f"Need {num_subjects}, but only {len(valid_subjects)} available. "
            f"Reduce num_steps or repetition_count."
        )

    # Randomly select subjects without replacement
    chosen_subjects = rng.sample(valid_subjects, num_subjects)
    print(f"  Selected {len(chosen_subjects)} unique subjects")

    # For each subject, select repetition_count triples
    edit_cases = []
    max_resample_attempts = 100

    for subject_idx, s in enumerate(chosen_subjects):
        # Calculate how many triples to select for this subject
        # For the last subject, use remaining edit cases
        remaining_edits = num_steps - len(edit_cases)
        current_rep_count = min(repetition_count, remaining_edits)

        # Get all triples for this subject
        available_triples = triples_by_s[s].copy()

        if len(available_triples) < current_rep_count:
            raise ValueError(
                f"Subject {s} has only {len(available_triples)} triples, "
                f"but needs {current_rep_count}"
            )

        # Randomly select current_rep_count triples from this subject
        selected_triples = rng.sample(available_triples, current_rep_count)

        # For each selected triple, find a valid o_new
        for triple in selected_triples:
            r = triple.r
            o_old = triple.o

            # Try to find valid o_new
            triple_found = False
            for attempt in range(max_resample_attempts):
                # Filter out o_old and existing objects for (s,r)
                existing_objects = sr2os.get((s, r), set())
                # Also exclude objects already used in edit cases for this (s,r)
                used_objects_this_sr = {
                    ec["o_new"] for ec in edit_cases
                    if ec["s"] == s and ec["r"] == r
                }
                cands = [
                    x for x in entities
                    if x != o_old
                    and x not in existing_objects
                    and x not in used_objects_this_sr
                ]

                if not cands:
                    # No valid o_new found, skip to next attempt
                    continue

                # Success: we found a valid o_new
                o_new = rng.choice(cands)
                edit_cases.append({
                    "s": s,
                    "r": r,
                    "o_old": o_old,
                    "o_new": o_new,
                    "repetition_count": repetition_count,
                    "seed": seed,
                })
                triple_found = True
                break

            if not triple_found:
                raise ValueError(
                    f"Could not find valid o_new for subject={s}, relation={r} "
                    f"after {max_resample_attempts} attempts"
                )

        if (subject_idx + 1) % 10 == 0 or subject_idx == len(chosen_subjects) - 1:
            print(f"  Progress: {subject_idx + 1}/{len(chosen_subjects)} subjects processed")

    print(f"  Generated {len(edit_cases)} edit cases")
    print(f"  Unique subjects used: {len(chosen_subjects)}")

    # Verification
    subject_counts = Counter(ec["s"] for ec in edit_cases)
    subjects_list = list(subject_counts.keys())

    # Check all subjects except possibly the last one
    for i, s in enumerate(subjects_list[:-1]):
        count = subject_counts[s]
        if count != repetition_count:
            raise ValueError(
                f"Subject {s} appears {count} times, expected {repetition_count}"
            )

    # Check last subject - it can have fewer repetitions
    if subjects_list:
        last_subject = subjects_list[-1]
        last_count = subject_counts[last_subject]
        if last_count > repetition_count:
            raise ValueError(
                f"Last subject {last_subject} appears {last_count} times, "
                f"expected at most {repetition_count}"
            )

    # Verify no collisions
    for case in edit_cases:
        assert case["o_new"] != case["o_old"], "o_new == o_old violation"
        existing = sr2os.get((case["s"], case["r"]), set())
        assert case["o_new"] not in existing, "o_new collision with existing triple"

    print(f"  ✓ All cases verified: each subject appears at most {repetition_count} times")
    print(f"  ✓ No collisions: o_new != o_old and no collisions with existing triples")

    return edit_cases


def main():
    """Main entry point for subject repetition editing."""
    parser = argparse.ArgumentParser(
        description="Run sequential editing with controlled subject repetition"
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
        help="Total number of sequential edits (last subject may have fewer repetitions if not divisible)",
    )
    parser.add_argument(
        "--repetition-count",
        type=int,
        required=True,
        choices=[1, 2, 4],
        help="Number of times each subject should be selected (1, 2, or 4)",
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
    print("Subject Repetition Sequential Editing")
    print("=" * 80)
    print(f"Model: {args.model_dir}")
    print(f"KG: {args.kg_dir}")
    print(f"Output: {args.output_dir}")
    print(f"NUM_STEPS: {args.num_steps}")
    print(f"REPETITION_COUNT: {args.repetition_count}")
    print(f"Number of subjects: {ceil(args.num_steps / args.repetition_count)}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    if args.num_steps % args.repetition_count != 0:
        last_subject_reps = args.num_steps % args.repetition_count
        print(f"⚠ Note: Last subject will have {last_subject_reps} repetition(s)")
    print("=" * 80)

    # Load KG
    print("\nLoading knowledge graph...")
    kg_path = Path(args.kg_dir) / "corpus.train.txt"
    kg = KG(str(kg_path))
    print(f"Loaded KG: {kg}")

    # Compute structures from KG
    print("\nComputing auxiliary structures...")
    triples_by_s, sr2os, entities = compute_structures_from_kg(kg)
    print(f"  Total subjects: {len(triples_by_s)}")
    print(f"  Total triples: {len(kg.triples)}")
    print(f"  Total entities: {len(entities)}")

    # Create config
    config = SeqEditConfig(
        model_dir=args.model_dir,
        kg_dir=args.kg_dir,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        num_retain_triples=args.num_retain_triples,
        eval_mode="sample",
        edit_selection_mode=f"repetition_{args.repetition_count}",
        edit_layer=args.layer,
        max_hop=args.max_hop,
        v_num_grad_steps=args.v_num_grad_steps,
        seed=args.seed,
        device=args.device,
    )

    # Add metadata to config
    config.repetition_mode = True
    config.repetition_count = args.repetition_count

    # Monkey-patch the sample_edit_cases function
    import src.sequential_edit.runner as runner_module

    original_sample_edit_cases = runner_module.sample_edit_cases

    def patched_sample_edit_cases(kg, num_cases, seed, selection_mode, max_hop):
        """Patched version that uses subject repetition sampling."""
        if selection_mode.startswith("repetition_"):
            return sample_edit_cases_with_repetition(
                kg,
                num_cases,
                args.repetition_count,
                triples_by_s,
                sr2os,
                entities,
                seed,
            )
        else:
            return original_sample_edit_cases(
                kg, num_cases, seed, selection_mode, max_hop
            )

    runner_module.sample_edit_cases = patched_sample_edit_cases

    # Run sequential edits
    try:
        run_sequential_edits(config)
        print("\n✓ Subject repetition editing completed successfully!")
        print(f"\nConfiguration summary:")
        print(f"  Total edit cases: {args.num_steps}")
        print(f"  Repetition count: {args.repetition_count}")
        print(f"  Number of unique subjects: {ceil(args.num_steps / args.repetition_count)}")
        if args.num_steps % args.repetition_count != 0:
            last_subject_reps = args.num_steps % args.repetition_count
            print(f"  Last subject repetitions: {last_subject_reps}")
        print(f"  Seed: {args.seed}")
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
