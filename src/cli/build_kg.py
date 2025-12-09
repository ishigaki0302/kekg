"""CLI script for building knowledge graphs."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import load_yaml, save_yaml, save_jsonl, set_seed, Logger
from src.kg import generate_graph, assign_synonyms, export_kg_corpus


def main():
    parser = argparse.ArgumentParser(description="Build knowledge graph")
    parser.add_argument("--config", type=str, required=True, help="Config YAML file")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    args = parser.parse_args()

    # Load config
    config = load_yaml(args.config)
    logger = Logger(verbose=True)

    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)

    logger.info(f"Building knowledge graph with config: {args.config}")

    # Extract KG config
    kg_config = config["kg"]
    graph_type = kg_config["type"]
    num_entities = kg_config["num_entities"]
    num_relations = kg_config["num_relations"]
    synonyms_per_entity = kg_config.get("synonyms_per_entity", 5)
    synonyms_per_relation = kg_config.get("synonyms_per_relation", 5)

    # Output directory
    output_dir = args.output_dir or config.get("output_dir", f"data/kg/{graph_type.lower()}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {graph_type} graph...")
    logger.info(f"  Entities: {num_entities}")
    logger.info(f"  Relations: {num_relations}")

    # Generate graph
    kg_kwargs = {
        "seed": seed
    }

    if graph_type == "ER":
        kg_kwargs["edge_prob"] = kg_config.get("edge_prob", 0.05)
    elif graph_type == "BA":
        kg_kwargs["m_attach"] = kg_config.get("m_attach", 3)

    kg = generate_graph(
        kind=graph_type,
        num_entities=num_entities,
        num_relations=num_relations,
        **kg_kwargs
    )

    logger.info(f"Generated {len(kg.triples)} triples")

    # Assign synonyms
    logger.info("Assigning synonyms...")
    kg = assign_synonyms(
        kg,
        synonyms_per_entity=synonyms_per_entity,
        synonyms_per_relation=synonyms_per_relation,
        seed=seed
    )

    # Save graph (JSONL format)
    graph_path = output_dir / "graph.jsonl"
    save_jsonl(kg.to_jsonl(), graph_path)
    logger.info(f"Saved graph to {graph_path}")

    # Save aliases
    aliases_path = output_dir / "aliases.json"
    aliases_data = {
        "entities": kg.entity_aliases,
        "relations": kg.relation_aliases
    }
    save_yaml(aliases_data, aliases_path)
    logger.info(f"Saved aliases to {aliases_path}")

    # Export corpus
    logger.info("Exporting corpus...")
    corpus_config = config.get("corpus", {})
    train_sample_rate = corpus_config.get("sample_rate_paraphrase", 0.2)

    corpus_info = export_kg_corpus(
        kg,
        output_dir=output_dir,
        train_sample_rate=train_sample_rate,
        seed=seed
    )

    logger.info(f"Corpus files:")
    logger.info(f"  Train: {corpus_info['train']} ({corpus_info['num_train']} samples)")
    logger.info(f"  All: {corpus_info['all']} ({corpus_info['num_all']} samples)")
    logger.info(f"  Base: {corpus_info['base']} ({corpus_info['num_base']} triples)")

    # Save metadata
    metadata = {
        "graph_type": graph_type,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "num_triples": len(kg.triples),
        "synonyms_per_entity": synonyms_per_entity,
        "synonyms_per_relation": synonyms_per_relation,
        "train_sample_rate": train_sample_rate,
        "seed": seed,
        "config": config
    }
    save_yaml(metadata, output_dir / "metadata.yaml")

    logger.info("Done!")


if __name__ == "__main__":
    main()
