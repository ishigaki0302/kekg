"""Continuous Knowledge Editing (CKE) protocol."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from copy import deepcopy

from ..modeling import GPTMini, SROInference, SROTokenizer
from ..edit import ROME, EditSpec
from ..kg import KnowledgeGraph


@dataclass
class CKEStep:
    """Single step in CKE sequence."""
    step: int
    edit_spec: EditSpec
    relation_type: str  # "shared" or "exclusive"

    # Post-edit metrics
    local_acc: float = 0.0
    local_rank: int = 0
    local_logp: float = 0.0

    # Retention metrics (for shared relations)
    retention_acc: float = 0.0
    retention_rank: float = 0.0

    # Overwrite metrics (for exclusive relations)
    overwrite_correct: bool = False  # Is latest target top-1?

    # Ripple metrics
    ripple_acc_delta: Dict[int, float] = field(default_factory=dict)


@dataclass
class CKEScenario:
    """CKE scenario configuration."""
    subject: str
    edits: List[Tuple[str, str, str]]  # List of (r, o, relation_type)
    order: str  # "fixed" or "shuffle"
    condition: str  # "entity_sim_high", "relation_sim_low", etc.


@dataclass
class CKEReport:
    """CKE experiment report."""
    scenario: CKEScenario
    steps: List[CKEStep]

    # Aggregate metrics
    final_retention_acc: float = 0.0
    final_overwrite_acc: float = 0.0
    plasticity_score: float = 0.0  # Avg local success
    stability_score: float = 0.0  # Avg retention


class CKEEvaluator:
    """Evaluator for Continuous Knowledge Editing."""

    def __init__(
        self,
        model: GPTMini,
        tokenizer: SROTokenizer,
        kg: KnowledgeGraph,
        device: str = "cuda"
    ):
        """
        Initialize CKE evaluator.

        Args:
            model: Base model (will be copied for each scenario)
            tokenizer: Tokenizer
            kg: Knowledge graph
            device: Device to run on
        """
        self.base_model = model
        self.tokenizer = tokenizer
        self.kg = kg
        self.device = device

    def run_scenario(
        self,
        scenario: CKEScenario,
        layers: List[int],
        **rome_kwargs
    ) -> CKEReport:
        """
        Run a single CKE scenario.

        Args:
            scenario: CKE scenario configuration
            layers: Layers to edit with ROME
            **rome_kwargs: Additional ROME parameters

        Returns:
            CKE report
        """
        # Copy model for this scenario
        model = deepcopy(self.base_model)
        rome = ROME(model, self.tokenizer, self.device)
        inference = SROInference(model, self.tokenizer, self.device)

        # Prepare edit sequence
        edits = scenario.edits.copy()
        if scenario.order == "shuffle":
            np.random.shuffle(edits)

        # Track state
        steps = []
        shared_edits = []  # Track shared-type edits for retention
        exclusive_history = {}  # Track exclusive-type latest target

        for step_idx, (r, o, rel_type) in enumerate(edits):
            # Create edit spec
            edit_spec = EditSpec(
                s=scenario.subject,
                r=r,
                o_old=None,
                o_new=o,
                edit_type="modify"
            )

            # Apply edit
            rome.apply_edit(edit_spec, layers, **rome_kwargs)

            # Evaluate local success
            local_eval = inference.evaluate_triple(scenario.subject, r, o)

            # Retention: evaluate all previous shared edits
            retention_acc = 0.0
            retention_rank = 0.0
            if shared_edits:
                retention_results = []
                for prev_r, prev_o in shared_edits:
                    result = inference.evaluate_triple(scenario.subject, prev_r, prev_o)
                    retention_results.append(result)

                retention_acc = np.mean([r["is_correct"] for r in retention_results])
                retention_rank = np.mean([r["rank"] for r in retention_results])

            # Overwrite: for exclusive relations, check if latest is top-1
            overwrite_correct = False
            if rel_type == "exclusive":
                if r in exclusive_history:
                    # Check if current target is top-1 (old should be displaced)
                    overwrite_correct = local_eval["is_correct"]
                exclusive_history[r] = o
            else:
                # For shared, just track correctness
                overwrite_correct = local_eval["is_correct"]

            # Ripple: compute hop-based impact (simplified)
            # For efficiency, we skip full ripple computation here
            # In practice, you would compute this per scenario
            ripple_acc_delta = {}

            # Record step
            step = CKEStep(
                step=step_idx,
                edit_spec=edit_spec,
                relation_type=rel_type,
                local_acc=float(local_eval["is_correct"]),
                local_rank=local_eval["rank"],
                local_logp=local_eval["log_probability"],
                retention_acc=retention_acc,
                retention_rank=retention_rank,
                overwrite_correct=overwrite_correct,
                ripple_acc_delta=ripple_acc_delta
            )
            steps.append(step)

            # Update tracking
            if rel_type == "shared":
                shared_edits.append((r, o))

        # Compute aggregate metrics
        final_retention = steps[-1].retention_acc if steps else 0.0

        # Final overwrite: check all exclusive relations have latest target
        final_overwrite = 0.0
        if exclusive_history:
            overwrite_checks = []
            for r, o_latest in exclusive_history.items():
                result = inference.evaluate_triple(scenario.subject, r, o_latest)
                overwrite_checks.append(result["is_correct"])
            final_overwrite = np.mean(overwrite_checks)

        # Plasticity: avg local success
        plasticity = np.mean([s.local_acc for s in steps])

        # Stability: avg retention across steps (excluding first step)
        stability_scores = [s.retention_acc for s in steps if s.step > 0]
        stability = np.mean(stability_scores) if stability_scores else 0.0

        report = CKEReport(
            scenario=scenario,
            steps=steps,
            final_retention_acc=final_retention,
            final_overwrite_acc=final_overwrite,
            plasticity_score=plasticity,
            stability_score=stability
        )

        return report

    def run_scenarios(
        self,
        scenarios: List[CKEScenario],
        layers: List[int],
        **rome_kwargs
    ) -> List[CKEReport]:
        """
        Run multiple CKE scenarios.

        Args:
            scenarios: List of scenarios
            layers: Layers to edit
            **rome_kwargs: ROME parameters

        Returns:
            List of reports
        """
        reports = []
        for scenario in scenarios:
            report = self.run_scenario(scenario, layers, **rome_kwargs)
            reports.append(report)
        return reports


def create_scenario_from_entity(
    kg: KnowledgeGraph,
    entity: str,
    num_steps: int = 5,
    relation_split: str = "mixed",  # "shared", "exclusive", "mixed"
    order: str = "fixed",
    condition: str = "default",
    seed: int = 42
) -> CKEScenario:
    """
    Create a CKE scenario from a knowledge graph entity.

    Args:
        kg: Knowledge graph
        entity: Subject entity
        num_steps: Number of editing steps
        relation_split: How to split shared/exclusive relations
        order: "fixed" or "shuffle"
        condition: Scenario condition name
        seed: Random seed

    Returns:
        CKE scenario
    """
    rng = np.random.default_rng(seed)

    # Find all triples with this entity as subject
    entity_triples = [t for t in kg.triples if t.s == entity]

    if not entity_triples:
        raise ValueError(f"No triples found for entity {entity}")

    # Sample triples
    if len(entity_triples) > num_steps:
        sampled = rng.choice(entity_triples, size=num_steps, replace=False)
    else:
        sampled = entity_triples[:num_steps]

    # Assign relation types
    edits = []
    for triple in sampled:
        # Use first alias
        r = kg.relation_aliases[triple.r][0]
        o = kg.entity_aliases[triple.o][0]

        # Assign type
        if relation_split == "shared":
            rel_type = "shared"
        elif relation_split == "exclusive":
            rel_type = "exclusive"
        else:  # mixed
            rel_type = rng.choice(["shared", "exclusive"])

        edits.append((r, o, rel_type))

    return CKEScenario(
        subject=kg.entity_aliases[entity][0],
        edits=edits,
        order=order,
        condition=condition
    )
