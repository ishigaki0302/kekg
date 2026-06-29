"""Pipeline: Generate ER graph, train model, run shattering experiments.

This is the control experiment to verify that our findings are due to
the BA graph's power-law degree distribution, not just graph structure per se.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_cmd(cmd, desc=""):
    print(f"\n{'='*60}")
    print(f"Step: {desc}")
    print(f"Cmd: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"FAILED: {desc}")
        sys.exit(1)
    print(f"Done: {desc}")


def main():
    python = sys.executable

    # Step 1: Generate ER graph (same entity/relation counts as BA)
    print("\n" + "=" * 60)
    print("Step 1: Generate ER Knowledge Graph")
    print("=" * 60)

    from src.kg.generator import generate_er_kg, write_txt, write_jsonl

    er_dir = Path("data/kg/er")
    er_dir.mkdir(parents=True, exist_ok=True)

    triples = generate_er_kg(
        num_entities=1200,
        num_relations=250,
        target_triples=30000,
        seed=42,
    )
    write_txt(triples, er_dir / "corpus.train.txt")
    write_jsonl(triples, er_dir / "graph.jsonl")

    # Print degree distribution comparison
    from collections import Counter
    degree = Counter()
    for t in triples:
        degree[t.s] += 1

    degrees = list(degree.values())
    import numpy as np
    print(f"\nER Graph Stats:")
    print(f"  Triples: {len(triples)}")
    print(f"  Entities with outgoing edges: {len(degree)}")
    print(f"  Degree: min={min(degrees)}, max={max(degrees)}, "
          f"mean={np.mean(degrees):.1f}, std={np.std(degrees):.1f}")

    # Step 2: Train model on ER graph
    run_cmd(
        [python, "src/cli/train_lm.py",
         "--config", "configs/train_gpt_small_er.yaml"],
        desc="Train GPT-small on ER graph",
    )

    # Step 3: Run Exp 1b (shattering) on ER model
    run_cmd(
        [python, "src/scripts/exp1b_degree_shattering.py",
         "--model-dir", "outputs/models/gpt_small_er",
         "--kg-corpus", "data/kg/er/corpus.train.txt",
         "--device", "cuda:0",
         "--output-dir", "outputs/exp1b_degree_shattering_er",
         "--num-edits-per-bin", "50",
         "--num-test-triples", "3000"],
        desc="Exp 1b: Degree-Conditional Shattering (ER)",
    )

    # Step 4: Run Exp 4 (ROME geometry) on ER model
    run_cmd(
        [python, "src/scripts/exp4_rome_geometry.py",
         "--model-dir", "outputs/models/gpt_small_er",
         "--kg-corpus", "data/kg/er/corpus.train.txt",
         "--device", "cuda:0",
         "--output-dir", "outputs/exp4_rome_geometry_er",
         "--num-edits-per-bin", "30"],
        desc="Exp 4: ROME Geometry (ER)",
    )

    print("\n" + "=" * 60)
    print("ER Control Pipeline Complete!")
    print("=" * 60)
    print("\nCompare results:")
    print("  BA shattering: outputs/exp1b_degree_shattering/")
    print("  ER shattering: outputs/exp1b_degree_shattering_er/")
    print("  BA geometry:   outputs/exp4_rome_geometry/")
    print("  ER geometry:   outputs/exp4_rome_geometry_er/")


if __name__ == "__main__":
    main()
