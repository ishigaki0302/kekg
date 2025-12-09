"""CLI script for running continuous knowledge editing experiments."""

import argparse
from pathlib import Path
import sys
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import load_yaml, save_yaml, set_seed, Logger, load_jsonl
from src.modeling import SROTokenizer, GPTMini, GPTConfig
from src.kg import KnowledgeGraph, Triple
from src.eval import CKEEvaluator, create_scenario_from_entity
from src.eval.viz import (
    plot_cke_rank_histogram,
    plot_cke_step_progression,
    plot_cke_heatmap,
    plot_order_effect_boxplot
)


def main():
    parser = argparse.ArgumentParser(description="Run CKE experiments")
    parser.add_argument("--config", type=str, required=True, help="Config YAML file")
    parser.add_argument("--model-dir", type=str, required=True, help="Trained model directory")
    parser.add_argument("--kg-dir", type=str, required=True, help="Knowledge graph directory")
    parser.add_argument("--num-scenarios", type=int, default=10, help="Number of scenarios per condition")
    parser.add_argument("--steps", type=int, default=5, help="Number of edit steps")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    args = parser.parse_args()

    # Load config
    config = load_yaml(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    # Output
    output_dir = Path(args.output_dir or "outputs/cke")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(output_dir / "cke.log", verbose=True)
    logger.info("Running CKE experiments")

    # Load tokenizer
    tokenizer_path = Path(args.model_dir) / "tokenizer.json"
    tokenizer = SROTokenizer.load(tokenizer_path)
    logger.info(f"Loaded tokenizer: vocab_size={tokenizer.vocab_size}")

    # Load model
    logger.info("Loading model...")
    model_path = Path(args.model_dir) / "model.pt"
    train_config = load_yaml(Path(args.model_dir) / "train_report.yaml")

    model_cfg = train_config["config"]["model"]
    gpt_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        d_model=model_cfg["d_model"],
        d_mlp=model_cfg["d_mlp"],
        max_seq_len=model_cfg.get("max_seq_len", 8),
        dropout=0.0  # No dropout for inference
    )

    model = GPTMini(gpt_config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    logger.info(f"Loaded model: {model.get_num_params():,} params")

    # Load KG
    logger.info("Loading knowledge graph...")
    kg_path = Path(args.kg_dir) / "graph.jsonl"
    aliases_path = Path(args.kg_dir) / "aliases.json"

    triples_data = load_jsonl(kg_path)
    triples = [Triple.from_dict(d) for d in triples_data]

    aliases = load_yaml(aliases_path)

    kg = KnowledgeGraph(
        entities=list(aliases["entities"].keys()),
        relations=list(aliases["relations"].keys()),
        triples=triples,
        entity_aliases=aliases["entities"],
        relation_aliases=aliases["relations"]
    )

    logger.info(f"Loaded KG: {len(kg.entities)} entities, {len(kg.triples)} triples")

    # Create evaluator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = CKEEvaluator(model, tokenizer, kg, device=device)

    # Edit layers (from config or default)
    edit_config = config.get("edit", {})
    layers = edit_config.get("layers", [0, 1, 2])
    logger.info(f"Edit layers: {layers}")

    # Generate scenarios
    logger.info(f"Generating {args.num_scenarios} scenarios...")

    conditions = [
        ("shared", "fixed", "shared_fixed"),
        ("exclusive", "fixed", "exclusive_fixed"),
        ("mixed", "fixed", "mixed_fixed"),
        ("mixed", "shuffle", "mixed_shuffle"),
    ]

    all_reports = []

    for relation_split, order, condition_name in conditions:
        logger.info(f"Running condition: {condition_name}")

        condition_reports = []

        # Sample entities
        rng = np.random.default_rng(seed)
        sampled_entities = rng.choice(kg.entities, size=min(args.num_scenarios, len(kg.entities)), replace=False)

        for i, entity in enumerate(sampled_entities):
            try:
                scenario = create_scenario_from_entity(
                    kg=kg,
                    entity=entity,
                    num_steps=args.steps,
                    relation_split=relation_split,
                    order=order,
                    condition=condition_name,
                    seed=seed + i
                )

                # Run scenario
                report = evaluator.run_scenario(scenario, layers)
                condition_reports.append(report)

                logger.info(
                    f"  Scenario {i+1}/{len(sampled_entities)}: "
                    f"Plasticity={report.plasticity_score:.3f}, "
                    f"Stability={report.stability_score:.3f}"
                )

            except Exception as e:
                logger.warning(f"  Scenario {i+1} failed: {e}")

        all_reports.extend(condition_reports)

        # Save condition reports
        condition_data = {
            "condition": condition_name,
            "num_scenarios": len(condition_reports),
            "avg_plasticity": np.mean([r.plasticity_score for r in condition_reports]),
            "avg_stability": np.mean([r.stability_score for r in condition_reports]),
            "avg_retention": np.mean([r.final_retention_acc for r in condition_reports]),
            "avg_overwrite": np.mean([r.final_overwrite_acc for r in condition_reports])
        }

        logger.info(f"  Condition summary: {condition_data}")
        save_yaml(condition_data, output_dir / f"{condition_name}_summary.yaml")

    # Generate visualizations
    logger.info("Generating visualizations...")

    # Group reports by condition
    reports_by_condition = {}
    for report in all_reports:
        cond = report.scenario.condition
        if cond not in reports_by_condition:
            reports_by_condition[cond] = []
        reports_by_condition[cond].append(report)

    # Rank histogram
    plot_cke_rank_histogram(
        all_reports[:4],  # First 4 reports
        output_dir / "figures" / "cke_rank_histogram.png"
    )

    # Step progression
    plot_cke_step_progression(
        [reports_by_condition[c][0] for c in ["shared_fixed", "exclusive_fixed", "mixed_fixed"] if c in reports_by_condition],
        output_dir / "figures" / "cke_step_progression.png"
    )

    # Heatmap for first report
    if all_reports:
        plot_cke_heatmap(
            all_reports[0],
            output_dir / "figures" / "cke_heatmap_example.png"
        )

    # Order effect
    if "mixed_fixed" in reports_by_condition and "mixed_shuffle" in reports_by_condition:
        plot_order_effect_boxplot(
            reports_by_condition["mixed_fixed"],
            reports_by_condition["mixed_shuffle"],
            output_dir / "figures" / "order_effect.png"
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
