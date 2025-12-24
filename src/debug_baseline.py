"""Debug script to check baseline accuracy."""

import sys
from pathlib import Path
import torch
import random

sys.path.insert(0, str(Path(__file__).parent))

from src.modeling import SROTokenizer, GPTMini, GPTConfig
from src.utils import load_yaml
from src.sequential_edit.kg_utils import KG

# Load model
model_dir = Path("outputs/models/gpt_small")
tokenizer = SROTokenizer.load(model_dir / "tokenizer.json")
train_report = load_yaml(model_dir / "train_report.yaml")
model_cfg = train_report["config"]["model"]

gpt_config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    n_layers=model_cfg["n_layers"],
    n_heads=model_cfg["n_heads"],
    d_model=model_cfg["d_model"],
    d_mlp=model_cfg["d_mlp"],
    max_seq_len=model_cfg.get("max_seq_len", 8),
    dropout=0.0
)

model = GPTMini(gpt_config)
model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Load KG from training corpus
kg_path = Path("data/kg/ba") / "corpus.train.txt"
kg = KG(str(kg_path))

# Sample triples
random.seed(42)
eval_triples = kg.sample_triples(20, seed=42)

print(f"Model: {model_cfg['n_layers']} layers")
print(f"KG: {kg}")
print(f"Evaluating {len(eval_triples)} triples...")
print("=" * 80)

correct = 0
for i, triple in enumerate(eval_triples, 1):
    # Create input
    eval_input = f"{triple.s} {triple.r}"
    input_ids = tokenizer.encode(eval_input)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Get prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs["logits"][0, -1, :]
        pred_id = logits.argmax().item()
        pred = tokenizer.get_token(pred_id)

    is_correct = pred == triple.o
    if is_correct:
        correct += 1

    # Show first 10 examples
    if i <= 10:
        print(f"{i}. Input: '{eval_input}'")
        print(f"   Expected: {triple.o}")
        print(f"   Predicted: {pred}")
        print(f"   {'✓ CORRECT' if is_correct else '✗ WRONG'}")

        # Show top 5 predictions
        top5_ids = logits.topk(5).indices.tolist()
        top5_probs = torch.softmax(logits, dim=0)[top5_ids].tolist()
        print(f"   Top 5 predictions:")
        for rank, (tid, prob) in enumerate(zip(top5_ids, top5_probs), 1):
            token = tokenizer.get_token(tid)
            marker = "←" if token == triple.o else ""
            print(f"     {rank}. {token}: {prob:.4f} {marker}")
        print()

print("=" * 80)
print(f"Accuracy: {correct}/{len(eval_triples)} = {correct/len(eval_triples):.2%}")
