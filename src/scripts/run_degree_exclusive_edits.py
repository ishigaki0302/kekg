"""Run sequential editing with exclusive (non-overlapping) subjects for degree high/low.

This script ensures that subject entities used in degree_high and degree_low
experiments do not overlap. Each subject is selected at most once, creating
exactly one edit case per subject.

Key features:
- Degree is calculated as out-degree from KG (not using pre-computed triple.degree_s)
- Extreme pools are created using percentile thresholds (with tie handling)
- Subjects are sampled exclusively (no repetition)
- Each subject contributes exactly one edit case
- Object collision avoidance: o_new is guaranteed not to exist in (s,r) existing objects

Usage:
    python -m src.scripts.run_degree_exclusive_edits \
        --model-dir outputs/models/gpt_small_no_alias \
        --kg-dir data/kg/ba_no_alias \
        --output-dir outputs/degree_exclusive/degree_high \
        --num-steps 1000 \
        --selection-mode degree_high \
        --num-retain-triples 1000 \
        --p-extreme 0.10 \
        --object-sampling global_entities \
        --collision-policy strict \
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


def compute_degree_from_kg(kg: KG) -> Tuple[Counter, Dict[str, List], Dict[Tuple[str, str], Set[str]], Dict[str, Set[str]], Set[str]]:
    """Compute out-degree and auxiliary structures from KG.

    Args:
        kg: Knowledge graph

    Returns:
        Tuple of:
            - deg: Counter of subject out-degrees
            - triples_by_s: Dict mapping subject to list of triples
            - sr2os: Dict mapping (s,r) to set of existing objects
            - objects_by_r: Dict mapping relation to set of objects
            - entities: Set of all entities (subjects + objects)
    """
    deg = Counter(t.s for t in kg.triples)

    triples_by_s = defaultdict(list)
    sr2os = defaultdict(set)
    objects_by_r = defaultdict(set)
    subjects = set()
    objects = set()

    for t in kg.triples:
        triples_by_s[t.s].append(t)
        sr2os[(t.s, t.r)].add(t.o)
        objects_by_r[t.r].add(t.o)
        subjects.add(t.s)
        objects.add(t.o)

    entities = subjects | objects

    return deg, dict(triples_by_s), dict(sr2os), dict(objects_by_r), entities


def get_extreme_pool(
    deg: Counter,
    p_extreme: float,
    mode: str
) -> Tuple[Set[str], int]:
    """Create extreme pool with tie handling.

    Args:
        deg: Counter of subject out-degrees
        p_extreme: Percentile threshold (e.g., 0.10 for top/bottom 10%)
        mode: "degree_high" or "degree_low"

    Returns:
        Tuple of (pool, threshold)
    """
    # Sort subjects by degree (descending)
    sorted_subjects = sorted(deg.keys(), key=lambda s: deg[s], reverse=True)
    n = len(sorted_subjects)
    k = ceil(p_extreme * n)

    if mode == "degree_high":
        # Top k subjects
        threshold = deg[sorted_subjects[k - 1]]
        pool = {s for s in sorted_subjects if deg[s] >= threshold}
    elif mode == "degree_low":
        # Bottom k subjects
        threshold = deg[sorted_subjects[-k]]
        # Exclude subjects in high pool to ensure exclusivity
        high_threshold = deg[sorted_subjects[k - 1]]
        high_pool = {s for s in sorted_subjects if deg[s] >= high_threshold}
        pool = {s for s in sorted_subjects if deg[s] <= threshold and s not in high_pool}
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return pool, threshold


def get_exclusive_subject_ranges(
    kg: KG,
    num_steps: int = None,
    p_extreme: float = 0.10
) -> Dict[str, Any]:
    """Compute exclusive subject pools for degree high and low using percentile-based thresholds.

    Args:
        kg: Knowledge graph
        num_steps: Requested number of steps per mode (if None, use entire pool)
        p_extreme: Percentile threshold (default: 0.10 for top/bottom 10%)

    Returns:
        Dictionary containing:
            - deg: Subject degree counter
            - triples_by_s: Subject to triples mapping
            - sr2os: (s,r) to existing objects mapping
            - objects_by_r: Relation to objects mapping
            - entities: All entities set
            - pool_high: Set of high-degree subjects
            - pool_low: Set of low-degree subjects
            - threshold_high: High degree threshold
            - threshold_low: Low degree threshold
    """
    # Compute degree from KG (not using triple.degree_s)
    deg, triples_by_s, sr2os, objects_by_r, entities = compute_degree_from_kg(kg)

    print(f"\nDegree computation:")
    print(f"  Total subjects: {len(deg)}")
    print(f"  Total triples: {len(kg.triples)}")
    print(f"  Total entities: {len(entities)}")

    # Create extreme pools
    pool_high, threshold_high = get_extreme_pool(deg, p_extreme, "degree_high")
    pool_low, threshold_low = get_extreme_pool(deg, p_extreme, "degree_low")

    print(f"\nExtreme pools (p_extreme={p_extreme}):")
    print(f"  High pool: {len(pool_high)} subjects, threshold={threshold_high}")
    print(f"  Low pool:  {len(pool_low)} subjects, threshold={threshold_low}")

    # Verify no overlap
    overlap = pool_high & pool_low
    if overlap:
        raise ValueError(f"Subject overlap detected: {overlap}")
    print(f"  ✓ No overlap between high and low pools")

    # Check pool sizes vs num_steps (if specified)
    if num_steps is not None:
        print(f"\nRequested num_steps: {num_steps}")
        if len(pool_high) < num_steps:
            raise ValueError(
                f"High pool size ({len(pool_high)}) < num_steps ({num_steps}). "
                f"Decrease num_steps or increase p_extreme."
            )
        if len(pool_low) < num_steps:
            raise ValueError(
                f"Low pool size ({len(pool_low)}) < num_steps ({num_steps}). "
                f"Decrease num_steps or increase p_extreme."
            )
        print(f"  ✓ Both pools have sufficient subjects (>= {num_steps})")
    else:
        print(f"\nNum_steps not specified: will use entire pool size")
        print(f"  High pool num_steps: {len(pool_high)}")
        print(f"  Low pool num_steps:  {len(pool_low)}")

    return {
        "deg": deg,
        "triples_by_s": triples_by_s,
        "sr2os": sr2os,
        "objects_by_r": objects_by_r,
        "entities": entities,
        "pool_high": pool_high,
        "pool_low": pool_low,
        "threshold_high": threshold_high,
        "threshold_low": threshold_low,
    }


def sample_edit_cases_exclusive(
    kg: KG,
    num_cases: int,
    selection_mode: str,
    pool: Set[str],
    deg: Counter,
    triples_by_s: Dict[str, List],
    sr2os: Dict[Tuple[str, str], Set[str]],
    objects_by_r: Dict[str, Set[str]],
    entities: Set[str],
    seed: int = 42,
    object_sampling: str = "global_entities",
    collision_policy: str = "strict"
) -> List[Dict[str, Any]]:
    """Sample edit cases with exclusive subject selection.

    Each subject is selected at most once. For each selected subject:
    1. Randomly select one triple from that subject
    2. Select o_new that doesn't exist in (s,r) existing objects

    Args:
        kg: Knowledge graph
        num_cases: Number of edit cases to sample
        selection_mode: "degree_high" or "degree_low"
        pool: Set of allowed subject entities
        deg: Subject degree counter
        triples_by_s: Subject to triples mapping
        sr2os: (s,r) to existing objects mapping
        objects_by_r: Relation to objects mapping
        entities: All entities set
        seed: Random seed
        object_sampling: "global_entities" or "by_relation_vocab"
        collision_policy: "strict" or "resample"

    Returns:
        List of edit cases with fields:
            - s, r, o_old, o_new, degree_s, selection_mode, seed
    """
    rng = random.Random(seed)

    print(f"\n[{selection_mode}] Sampling edit cases:")
    print(f"  Pool size: {len(pool)}")
    print(f"  Requested cases: {num_cases}")
    print(f"  Object sampling: {object_sampling}")
    print(f"  Collision policy: {collision_policy}")

    # Step 3: Sample subjects without replacement
    if len(pool) < num_cases:
        if collision_policy == "strict":
            raise ValueError(
                f"Pool size ({len(pool)}) < num_cases ({num_cases}). "
                f"Cannot sample {num_cases} unique subjects."
            )
        else:
            print(f"  ⚠ Pool size ({len(pool)}) < num_cases ({num_cases}), using all pool")
            chosen_subjects = list(pool)
    else:
        chosen_subjects = rng.sample(list(pool), num_cases)

    print(f"  Chosen subjects: {len(chosen_subjects)}")

    # Log degree statistics
    chosen_degrees = [deg[s] for s in chosen_subjects]
    print(f"  Degree range: min={min(chosen_degrees)}, median={sorted(chosen_degrees)[len(chosen_degrees)//2]}, max={max(chosen_degrees)}")

    # Step 4-5: For each subject, select one triple and find o_new
    edit_cases = []
    max_resample_attempts = 10
    max_subject_resample_attempts = 100

    used_subjects = set()
    available_pool = set(pool)

    for subject_idx in range(len(chosen_subjects)):
        s = chosen_subjects[subject_idx]

        # Try to find a valid (triple, o_new) for this subject
        triple_found = False
        for attempt in range(max_resample_attempts):
            # Select one triple from this subject
            triple = rng.choice(triples_by_s[s])
            r = triple.r
            o_old = triple.o

            # Determine candidate objects
            if object_sampling == "by_relation_vocab":
                cand_base = objects_by_r.get(r, entities)
                if not cand_base:
                    cand_base = entities
            else:  # global_entities
                cand_base = entities

            # Filter out o_old and existing objects for (s,r)
            existing_objects = sr2os.get((s, r), set())
            cands = [x for x in cand_base if x != o_old and x not in existing_objects]

            if not cands:
                if collision_policy == "strict":
                    raise ValueError(
                        f"No valid o_new for subject={s}, relation={r}. "
                        f"All candidates are either o_old or exist in (s,r)."
                    )
                # resample: try another triple from the same subject
                continue
            else:
                # Success: we found a valid o_new
                o_new = rng.choice(cands)
                edit_cases.append({
                    "s": s,
                    "r": r,
                    "o_old": o_old,
                    "o_new": o_new,
                    "degree_s": deg[s],
                    "selection_mode": selection_mode,
                    "seed": seed,
                })
                used_subjects.add(s)
                triple_found = True
                break

        if not triple_found:
            # All triples for this subject failed to find valid o_new
            if collision_policy == "strict":
                raise ValueError(
                    f"No valid triple for subject={s} after {max_resample_attempts} attempts."
                )
            else:
                # resample: pick a new subject from available pool
                available_pool.discard(s)
                available_pool -= used_subjects

                for sub_attempt in range(max_subject_resample_attempts):
                    if not available_pool:
                        print(f"  ⚠ Warning: Available pool exhausted, stopping at {len(edit_cases)} cases")
                        break

                    new_s = rng.choice(list(available_pool))

                    # Try to find a valid triple for this new subject
                    inner_found = False
                    for inner_attempt in range(max_resample_attempts):
                        triple = rng.choice(triples_by_s[new_s])
                        r = triple.r
                        o_old = triple.o

                        if object_sampling == "by_relation_vocab":
                            cand_base = objects_by_r.get(r, entities)
                            if not cand_base:
                                cand_base = entities
                        else:
                            cand_base = entities

                        existing_objects = sr2os.get((new_s, r), set())
                        cands = [x for x in cand_base if x != o_old and x not in existing_objects]

                        if cands:
                            o_new = rng.choice(cands)
                            edit_cases.append({
                                "s": new_s,
                                "r": r,
                                "o_old": o_old,
                                "o_new": o_new,
                                "degree_s": deg[new_s],
                                "selection_mode": selection_mode,
                                "seed": seed,
                            })
                            used_subjects.add(new_s)
                            available_pool.discard(new_s)
                            inner_found = True
                            break

                    if inner_found:
                        break

                if not inner_found:
                    print(f"  ⚠ Warning: Could not find replacement subject after {max_subject_resample_attempts} attempts")

    print(f"  Generated {len(edit_cases)} edit cases")
    print(f"  Unique subjects used: {len(used_subjects)}")

    # Verification
    assert len(used_subjects) == len(edit_cases), "Subject uniqueness violation"
    for case in edit_cases:
        assert case["o_new"] != case["o_old"], "o_new == o_old violation"
        assert case["o_new"] not in sr2os.get((case["s"], case["r"]), set()), "o_new collision violation"

    print(f"  ✓ All cases verified: unique subjects, o_new != o_old, no collisions")

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
        default=None,
        help="Number of sequential edits (default: None = use entire pool determined by p-extreme)",
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
        "--p-extreme",
        type=float,
        default=0.10,
        help="Percentile threshold for extreme pool (default: 0.10 for top/bottom 10%%)",
    )
    parser.add_argument(
        "--object-sampling",
        type=str,
        default="global_entities",
        choices=["global_entities", "by_relation_vocab"],
        help="Object sampling mode (default: global_entities)",
    )
    parser.add_argument(
        "--collision-policy",
        type=str,
        default="strict",
        choices=["strict", "resample"],
        help="Collision handling policy (default: strict)",
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
    print(f"P_EXTREME: {args.p_extreme}")
    print(f"NUM_STEPS: {args.num_steps if args.num_steps is not None else 'auto (use entire pool)'}")
    print(f"Object sampling: {args.object_sampling}")
    print(f"Collision policy: {args.collision_policy}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Load KG
    print("\nLoading knowledge graph...")
    kg_path = Path(args.kg_dir) / "corpus.train.txt"
    kg = KG(str(kg_path))
    print(f"Loaded KG: {kg}")

    # Compute exclusive subject ranges
    ranges = get_exclusive_subject_ranges(kg, args.num_steps, args.p_extreme)

    # Extract structures
    deg = ranges["deg"]
    triples_by_s = ranges["triples_by_s"]
    sr2os = ranges["sr2os"]
    objects_by_r = ranges["objects_by_r"]
    entities = ranges["entities"]

    # Select appropriate pool based on mode
    if args.selection_mode == "degree_high":
        pool = ranges["pool_high"]
        threshold = ranges["threshold_high"]
    else:  # degree_low
        pool = ranges["pool_low"]
        threshold = ranges["threshold_low"]

    # If num_steps not specified, use entire pool
    actual_num_steps = args.num_steps if args.num_steps is not None else len(pool)

    print(f"\n[{args.selection_mode}] Selected pool:")
    print(f"  Pool size: {len(pool)}")
    print(f"  Threshold: {threshold}")
    print(f"  Actual num_steps: {actual_num_steps}")

    # Create config
    config = SeqEditConfig(
        model_dir=args.model_dir,
        kg_dir=args.kg_dir,
        output_dir=args.output_dir,
        num_steps=actual_num_steps,
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
    config.p_extreme = args.p_extreme
    config.object_sampling = args.object_sampling
    config.collision_policy = args.collision_policy
    config.pool_size = len(pool)
    config.threshold = threshold

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
                pool,
                deg,
                triples_by_s,
                sr2os,
                objects_by_r,
                entities,
                seed,
                args.object_sampling,
                args.collision_policy
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
        print(f"\nConfiguration summary:")
        print(f"  Mode: {args.selection_mode}")
        print(f"  Pool size: {len(pool)}")
        print(f"  Threshold: {threshold}")
        print(f"  Cases generated: {actual_num_steps}")
        print(f"  P_extreme: {args.p_extreme}")
        print(f"  Object sampling: {args.object_sampling}")
        print(f"  Collision policy: {args.collision_policy}")
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
