"""Export knowledge graph to corpus formats."""

from pathlib import Path
from typing import List, Union
from .generator import KnowledgeGraph, Triple, expand_triples_with_aliases


def triples_to_corpus(
    triples: List[Triple],
    output_path: Union[str, Path],
    use_canonical_output: bool = True
) -> None:
    """
    Convert triples to 3-token corpus format (space-separated SRO).

    Args:
        triples: List of triples
        output_path: Output file path
        use_canonical_output: If True, use canonical entity ID for output (no aliases)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for triple in triples:
            s = triple.s
            r = triple.r
            o = triple.o

            # Extract canonical ID from alias (remove __a* suffix)
            if use_canonical_output and '__a' in o:
                o = o.split('__a')[0]

            # Write as space-separated tokens
            f.write(f"{s} {r} {o}\n")


def export_kg_corpus(
    kg: KnowledgeGraph,
    output_dir: Union[str, Path],
    train_sample_rate: float = 0.2,
    seed: int = 42
) -> dict:
    """
    Export knowledge graph to corpus files.

    Creates two files:
    - corpus.train.txt: Sampled alias combinations (for training)
    - corpus.all.txt: All alias combinations (for evaluation)

    Args:
        kg: Knowledge graph with aliases
        output_dir: Output directory
        train_sample_rate: Sampling rate for training set
        seed: Random seed

    Returns:
        Dictionary with paths to generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate train set (sampled aliases)
    train_triples = expand_triples_with_aliases(
        kg, sample_rate=train_sample_rate, seed=seed
    )
    train_path = output_dir / "corpus.train.txt"
    triples_to_corpus(train_triples, train_path)

    # Generate full evaluation set (all aliases)
    all_triples = expand_triples_with_aliases(
        kg, sample_rate=1.0, seed=seed
    )
    all_path = output_dir / "corpus.all.txt"
    triples_to_corpus(all_triples, all_path)

    # Also save base triples (canonical form)
    base_path = output_dir / "corpus.base.txt"
    triples_to_corpus(kg.triples, base_path)

    return {
        "train": str(train_path),
        "all": str(all_path),
        "base": str(base_path),
        "num_train": len(train_triples),
        "num_all": len(all_triples),
        "num_base": len(kg.triples)
    }
