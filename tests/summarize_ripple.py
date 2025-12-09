#!/usr/bin/env python
"""Summarize ripple effect analysis results from batch editing."""

import json
import sys
from pathlib import Path

def summarize_edit(edit_num, base_dir):
    """Summarize a single edit result."""
    edit_dir = base_dir / f"edit_{edit_num}"
    ripple_file = edit_dir / "ripple_analysis.json"

    if not ripple_file.exists():
        print(f"Edit {edit_num}: No ripple analysis found")
        return

    with open(ripple_file, 'r') as f:
        data = json.load(f)

    edit_info = data['edit']
    stats = data['statistics']

    print(f"=== Edit {edit_num} ===")
    print(f"Edit: ({edit_info['s']}, {edit_info['r']}, {edit_info['o_original']}) -> {edit_info['o_target']}")
    print(f"Layer: {edit_info['layer']}")
    print(f"Total triples: {stats['total_triples']}")
    print(f"By hop distance:")

    for hop in sorted(stats['by_hop_distance'].keys(), key=int):
        hop_stats = stats['by_hop_distance'][hop]
        print(f"  Hop {hop}: {hop_stats['count']:4d} triples, "
              f"mean ripple = {hop_stats['mean_ripple']:.4f} ± {hop_stats['std_ripple']:.4f}")
    print()

def main():
    base_dir = Path("outputs/editing_batch_results")

    print("=" * 70)
    print("ROME Batch Editing - Ripple Effect Summary")
    print("=" * 70)
    print()

    for i in range(1, 11):
        summarize_edit(i, base_dir)

if __name__ == "__main__":
    main()
