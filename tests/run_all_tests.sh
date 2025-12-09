#!/bin/bash
# Run all tests

echo "=== Running SRO Platform Tests ==="
echo ""

echo "Test 1: Knowledge Graph Generation"
python tests/test_kg_generation.py
echo ""

echo "Test 2: Tokenizer"
python tests/test_tokenizer.py
echo ""

echo "Test 3: Model"
python tests/test_model.py
echo ""

echo "=== All tests complete! ==="
