"""Configuration for sequential editing experiments."""

from dataclasses import dataclass, field
from typing import Literal, Sequence, Tuple

EditMethod = Literal["rome"]  # Future: "memit", "mend", etc.
EditSelectionMode = Literal["random", "degree_high", "degree_low", "hop_high", "hop_low"]


@dataclass
class SeqEditConfig:
    """Configuration for sequential editing experiments.

    Attributes:
        num_steps: Number of sequential edits to perform
        num_eval_triples: Number of triples to use for ripple/KG accuracy evaluation
        max_hop: Maximum hop distance to compute for ripple effect analysis
        degree_bins: Degree bins for grouping entities (min, max) tuples
        edit_method: Knowledge editing method to use (currently only "rome")
        seed: Random seed for reproducibility
        output_dir: Output directory for results and visualizations
        model_dir: Directory containing trained model and tokenizer
        kg_dir: Directory containing knowledge graph data
        device: Device to run on ("cuda" or "cpu")
    """

    eval_mode: str = "all"
    # "sample"       … これまで通りサンプリング
    # "all"          … KG 全トリプル（edit case 含む）
    # "all_excl_edits" … KG 全トリプルのうち edit に使った triple を除く

    edit_selection_mode: EditSelectionMode = "random"
    # "random"       … ランダムサンプリング
    # "degree_high"  … Subjectの次数が大きいものを優先
    # "degree_low"   … Subjectの次数が小さいものを優先
    # "hop_high"     … Subjectからのhop数平均が大きいものを優先
    # "hop_low"      … Subjectからのhop数平均が小さいものを優先

    num_steps: int = 50
    num_eval_triples: int = 1000
    num_retain_triples: int = 1000  # 編集しない知識集合（保持評価用）の数
    max_hop: int = 10  # 通常は変更不要（小規模KGでは全ノードが到達可能）
    degree_bins: Sequence[Tuple[int, int]] = field(default_factory=lambda: [
        (0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 999)
    ])

    # Edit method configuration
    edit_method: EditMethod = "rome"
    edit_layer: int = None  # Layer to edit (None for auto-locate)
    v_num_grad_steps: int = 5  # Number of gradient steps for ROME right vector (reduced from 20 to prevent overfitting)

    # Paths
    model_dir: str = "outputs/models/gpt_mini_ba"
    kg_dir: str = "data/kg/ba"
    output_dir: str = "outputs/sequential"

    # Random seed
    seed: int = 42

    # Device
    device: str = "cuda"

    def to_dict(self):
        """Convert config to dictionary for JSON serialization."""
        return {
            "num_steps": self.num_steps,
            "num_eval_triples": self.num_eval_triples,
            "num_retain_triples": self.num_retain_triples,
            "max_hop": self.max_hop,
            "degree_bins": list(self.degree_bins),
            "edit_method": self.edit_method,
            "edit_selection_mode": self.edit_selection_mode,
            "edit_layer": self.edit_layer,
            "v_num_grad_steps": self.v_num_grad_steps,
            "model_dir": self.model_dir,
            "kg_dir": self.kg_dir,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "device": self.device,
            "eval_mode": self.eval_mode,
        }
