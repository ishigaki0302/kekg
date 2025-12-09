"""ROME (Rank-One Model Editing) implementation.

Based on the paper "Locating and Editing Factual Associations in GPT"
https://rome.baulab.info/

This implementation adapts ROME to work with our GPT mini model by importing
the reference implementation and bridging interface differences.
"""

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn

# Add ROME reference implementation to path
ROME_PATH = Path(__file__).parent / "knowledge_editing_method_repositorys" / "rome"
sys.path.insert(0, str(ROME_PATH))

from rome.rome_main import apply_rome_to_model, execute_rome
from rome.rome_hparams import ROMEHyperParams

from .tokenizer_wrapper import TokenizerWrapper
from .causal_tracing import CausalTracer


@dataclass
class EditSpec:
    """Specification for a knowledge edit."""

    s: str  # Subject
    r: str  # Relation
    o_target: str  # Target object


@dataclass
class EditResult:
    """Result of applying a ROME edit."""

    success: bool
    layer: int
    original_prediction: str
    new_prediction: str
    edit_spec: EditSpec


class ModelConfigWrapper:
    """Wrapper to add HF-style config attributes to our GPT model config."""

    def __init__(self, original_config):
        """
        Args:
            original_config: Original GPT model config
        """
        self.original_config = original_config

        # Copy all original attributes
        for attr in dir(original_config):
            if not attr.startswith("_"):
                setattr(self, attr, getattr(original_config, attr))

        # Add HF-style attribute aliases
        if hasattr(original_config, "d_model"):
            self.n_embd = original_config.d_model
        if hasattr(original_config, "n_layers"):
            self.n_layer = original_config.n_layers

        # Add model name for ROME caching
        self._name_or_path = "gpt_mini"


class ModelWrapper(nn.Module):
    """Wrapper to make GPT model compatible with reference ROME implementation."""

    def __init__(self, model):
        """
        Args:
            model: Original GPT model
        """
        super().__init__()
        object.__setattr__(self, 'model', model)
        object.__setattr__(self, 'config', ModelConfigWrapper(model.config))
        object.__setattr__(self, 'device', next(model.parameters()).device)

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        """
        Forward pass compatible with HF interface.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask (ignored for now)
            **kwargs: Other arguments (ignored)

        Returns:
            Object with .logits attribute
        """
        # Our model returns dict with "logits" key
        outputs = self.model(input_ids)

        # Wrap in object with .logits attribute for HF compatibility
        class OutputWrapper:
            def __init__(self, logits):
                self.logits = logits

        return OutputWrapper(outputs["logits"])

    def named_modules(self):
        """Pass through to original model."""
        return self.model.named_modules()

    def named_parameters(self):
        """Pass through to original model."""
        return self.model.named_parameters()

    def parameters(self):
        """Pass through to original model."""
        return self.model.parameters()

    def eval(self):
        """Pass through to original model."""
        return self.model.eval()

    def train(self, mode=True):
        """Pass through to original model."""
        return self.model.train(mode)

    def to(self, device):
        """Pass through to original model."""
        self.model = self.model.to(device)
        self.device = device
        return self

    def __getattr__(self, name):
        """Delegate unknown attributes to original model."""
        # Avoid infinite recursion by using object.__getattribute__
        try:
            model = object.__getattribute__(self, 'model')
            return getattr(model, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class ROME:
    """ROME knowledge editor."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda",
        kg_corpus_path: Optional[str] = None,
        mom2_n_samples: int = 10000,  # Use more samples from KG
        use_mom2_adjustment: bool = True,
        v_lr: float = 0.5,
        v_num_grad_steps: int = 20,
        v_weight_decay: float = 0.5,
        kl_factor: float = 0.0625,
        clamp_norm_factor: float = 4.0
    ):
        """
        Initialize ROME editor.

        Args:
            model: The language model to edit
            tokenizer: Tokenizer for the model
            device: Device to run on
            kg_corpus_path: Path to KG corpus for computing statistics
                           (should use trained KG data with all learned knowledge)
            mom2_n_samples: Number of samples for second moment statistics
                          (default 10000 to use full KG if available)
            use_mom2_adjustment: Whether to use second moment adjustment
            v_lr: Learning rate for computing right vector
            v_num_grad_steps: Number of gradient steps for right vector
            v_weight_decay: Weight decay for right vector optimization
            kl_factor: KL divergence weight
            clamp_norm_factor: Max norm for right vector
        """
        self.original_model = model
        self.model = ModelWrapper(model)
        self.original_tokenizer = tokenizer
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.device = device
        self.kg_corpus_path = kg_corpus_path
        self.use_mom2_adjustment = use_mom2_adjustment

        # Hyperparameters
        self.v_lr = v_lr
        self.v_num_grad_steps = v_num_grad_steps
        self.v_weight_decay = v_weight_decay
        self.kl_factor = kl_factor
        self.clamp_norm_factor = clamp_norm_factor
        self.mom2_n_samples = mom2_n_samples

        # Initialize causal tracer
        self.tracer = CausalTracer(model, tokenizer, device)

    def _create_hparams(self, layer: int) -> ROMEHyperParams:
        """
        Create ROME hyperparameters for our model.

        Args:
            layer: Layer to edit

        Returns:
            ROMEHyperParams instance
        """
        hparams_dict = {
            "layers": [layer],
            "fact_token": "subject_last",
            "v_num_grad_steps": self.v_num_grad_steps,
            "v_lr": self.v_lr,
            "v_loss_layer": layer,  # Use same layer for loss
            "v_weight_decay": self.v_weight_decay,
            "clamp_norm_factor": self.clamp_norm_factor,
            "kl_factor": self.kl_factor,
            "mom2_adjustment": self.use_mom2_adjustment,
            "context_template_length_params": [],  # Empty to avoid generation

            # Module templates for our GPT model structure
            "rewrite_module_tmp": "blocks.{}.ffn.w2",  # MLP output layer
            "layer_module_tmp": "blocks.{}",
            "mlp_module_tmp": "blocks.{}.ffn",
            "attn_module_tmp": "blocks.{}.attn",
            "ln_f_module": "ln_f",
            "lm_head_module": "lm_head",

            # Statistics
            "mom2_dataset": "custom",
            "mom2_n_samples": self.mom2_n_samples,
            "mom2_dtype": "float32"
        }

        return ROMEHyperParams(**hparams_dict)

    def locate_important_layer(
        self,
        s: str,
        r: str,
        o_target: str,
        noise_level: float = 3.0,
        num_samples: int = 10
    ) -> int:
        """
        Locate the layer where the fact is stored.

        Args:
            s: Subject
            r: Relation
            o_target: Target object
            noise_level: Noise level for causal tracing
            num_samples: Number of samples for tracing

        Returns:
            Layer index where fact is stored
        """
        return self.tracer.locate_important_layer(
            s, r, o_target, noise_level, num_samples
        )

    def locate_important_layer_with_scores(
        self,
        s: str,
        r: str,
        o_target: str,
        noise_level: float = 3.0,
        num_samples: int = 10
    ) -> Tuple[int, List[Dict], torch.Tensor]:
        """
        Locate important layer with detailed scores.

        Returns:
            Tuple of (best_layer, layer_effects, token_layer_grid)
        """
        return self.tracer.locate_important_layer_with_scores(
            s, r, o_target, noise_level, num_samples
        )

    def apply_edit(
        self,
        s: str,
        r: str,
        o_target: str,
        layer: Optional[int] = None,
        copy_model: bool = False
    ) -> Tuple[nn.Module, EditResult]:
        """
        Apply a ROME edit to the model.

        Args:
            s: Subject
            r: Relation
            o_target: Target object
            layer: Layer to edit (if None, automatically locate)
            copy_model: Whether to copy model before editing

        Returns:
            Tuple of (edited_model, edit_result)
        """
        # Get original prediction
        input_text = f"{s} {r}"
        input_ids = torch.tensor([self.original_tokenizer.encode(input_text)]).to(self.device)

        with torch.no_grad():
            outputs = self.original_model(input_ids)
            logits = outputs["logits"][0, -1, :]
            orig_pred_id = torch.argmax(logits).item()
            orig_pred = self.original_tokenizer.get_token(orig_pred_id)

        # Locate layer if not provided
        if layer is None:
            layer = self.locate_important_layer(s, r, o_target)

        # Create request for reference implementation
        # IMPORTANT: prompt must include relation to match actual usage
        request = {
            "prompt": f"{{}} {r}",  # Will be formatted with subject: "{subject} {relation}"
            "subject": s,
            "relation": r,
            "target_new": {
                "str": f" {o_target}"  # Add space for correct tokenization
            }
        }

        # Create hyperparameters
        hparams = self._create_hparams(layer)

        # Copy model if requested
        model_to_edit = ModelWrapper(deepcopy(self.original_model)) if copy_model else self.model

        # Apply ROME using reference implementation
        edited_model, _ = apply_rome_to_model(
            model_to_edit,
            self.tokenizer,
            [request],
            hparams,
            copy=False,  # We already copied if needed
            return_orig_weights=False
        )

        # Get the underlying model (unwrap)
        if isinstance(edited_model, ModelWrapper):
            result_model = edited_model.model
        else:
            result_model = edited_model

        # Get new prediction
        with torch.no_grad():
            outputs = result_model(input_ids)
            logits = outputs["logits"][0, -1, :]
            new_pred_id = torch.argmax(logits).item()
            new_pred = self.original_tokenizer.get_token(new_pred_id)

        result = EditResult(
            success=(new_pred.strip() == o_target.strip()),
            layer=layer,
            original_prediction=orig_pred.strip(),
            new_prediction=new_pred.strip(),
            edit_spec=EditSpec(s=s, r=r, o_target=o_target)
        )

        return result_model, result
