#!/usr/bin/env python
"""Check edit success rate."""

import json
from pathlib import Path

base_dir = Path("outputs/editing_batch_results")

print("Edit Success Analysis")
print("=" * 80)

success_count = 0
total_count = 10

for i in range(1, 11):
    result_file = base_dir / f"edit_{i}" / "edit_result.json"
    with open(result_file, 'r') as f:
        data = json.load(f)

    success = data['success']
    if success:
        success_count += 1

    print(f"Edit {i:2d}: ({data['s']}, {data['r']}, {data['o_original']}) -> {data['o_target']}")
    print(f"  Original Prediction: {data['original_prediction']}")
    print(f"  New Prediction:      {data['new_prediction']}")
    print(f"  Target:              {data['o_target']}")
    print(f"  Success:             {success}")

    # Check if prediction changed
    changed = data['original_prediction'] != data['new_prediction']
    print(f"  Prediction Changed:  {changed}")
    print()

print("=" * 80)
print(f"Success Rate: {success_count}/{total_count} = {success_count/total_count*100:.1f}%")
