"""Causal tracing for knowledge localization in neural networks.

Based on the ROME paper: https://rome.baulab.info/
This implementation adapts ROME's causal tracing to work with our GPT mini model.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TracingResult:
    """Result of causal tracing analysis."""

    scores: torch.Tensor  # [num_tokens, num_layers] indirect effects
    low_score: float  # Probability when all subject tokens are corrupted
    high_score: float  # Probability with clean input
    input_ids: torch.Tensor  # Input token IDs
    input_tokens: List[str]  # Input token strings
    subject_range: Tuple[int, int]  # (start, end) indices of subject tokens
    answer: str  # Predicted answer token
    correct_prediction: bool  # Whether model predicted correctly
    kind: Optional[str] = None  # "mlp", "attn", or None for all


class CausalTracer:
    """Performs causal tracing to locate where facts are stored in the model."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda"
    ):
        """
        Initialize causal tracer.

        Args:
            model: The language model to trace
            tokenizer: Tokenizer for the model
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_layers = model.config.n_layers

    def trace_important_states(
        self,
        s: str,
        r: str,
        o_target: str,
        noise_level: float = 3.0,
        num_samples: int = 10,
        kind: Optional[str] = None
    ) -> TracingResult:
        """
        Run causal tracing to find important hidden states.

        This function corrupts the subject tokens with noise, then restores
        individual hidden states one at a time to measure their causal effect
        on predicting the target object.

        Args:
            s: Subject string
            r: Relation string
            o_target: Target object string
            noise_level: Standard deviations of noise to add
            num_samples: Number of noise samples to average over
            kind: Type of component to trace - None (all), "mlp", or "attn"

        Returns:
            TracingResult containing the causal effects at each (token, layer)
        """
        # Create input
        input_text = f"{s} {r}"
        input_ids = torch.tensor([self.tokenizer.encode(input_text)]).to(self.device)

        # Get clean prediction
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs["logits"][0, -1, :]  # Last token logits
            probs = torch.softmax(logits, dim=-1)
            pred_token_id = torch.argmax(probs).item()
            pred_token = self.tokenizer.get_token(pred_token_id)

            # Get target token ID
            target_token_id = self.tokenizer.encode(o_target)[0]
            high_score = probs[target_token_id].item()

        # Check if prediction is correct
        correct_prediction = (pred_token.strip() == o_target.strip())

        # Find subject token range
        s_tokens = self.tokenizer.encode(s)
        subject_range = (0, len(s_tokens))

        # Get embedding std for noise
        embed_std = self._get_embedding_std()
        noise_std = noise_level * embed_std

        # Measure effect of full corruption
        low_score = self._trace_with_patch(
            input_ids=input_ids,
            states_to_patch=[],  # No restoration
            subject_range=subject_range,
            target_token_id=target_token_id,
            noise_std=noise_std,
            num_samples=num_samples
        )

        # Trace importance of each (token, layer) pair
        num_tokens = input_ids.shape[1]
        scores = torch.zeros(num_tokens, self.num_layers)

        for token_idx in range(num_tokens):
            for layer_idx in range(self.num_layers):
                # Restore this (token, layer) state
                score = self._trace_with_patch(
                    input_ids=input_ids,
                    states_to_patch=[(token_idx, layer_idx)],
                    subject_range=subject_range,
                    target_token_id=target_token_id,
                    noise_std=noise_std,
                    num_samples=num_samples,
                    kind=kind
                )
                scores[token_idx, layer_idx] = score

        # Get token strings
        token_strs = [self.tokenizer.get_token(tid) for tid in input_ids[0].tolist()]

        return TracingResult(
            scores=scores,
            low_score=low_score,
            high_score=high_score,
            input_ids=input_ids[0],
            input_tokens=token_strs,
            subject_range=subject_range,
            answer=pred_token,
            correct_prediction=correct_prediction,
            kind=kind
        )

    def _trace_with_patch(
        self,
        input_ids: torch.Tensor,
        states_to_patch: List[Tuple[int, int]],
        subject_range: Tuple[int, int],
        target_token_id: int,
        noise_std: float,
        num_samples: int,
        kind: Optional[str] = None
    ) -> float:
        """
        Run forward pass with noise and optional state restoration.

        Args:
            input_ids: Input token IDs [1, seq_len]
            states_to_patch: List of (token_idx, layer_idx) to restore
            subject_range: (start, end) of subject tokens to corrupt
            target_token_id: Token ID to measure probability of
            noise_std: Standard deviation of noise
            num_samples: Number of noise samples

        Returns:
            Average probability of target token across samples
        """
        # Create batch with clean sample followed by noisy samples
        batch_size = num_samples + 1
        input_ids_batch = input_ids.repeat(batch_size, 1)

        # Store clean hidden states at each layer
        clean_states = {}

        def make_hook(layer_idx: int, is_clean_pass: bool, component: str = "block"):
            """Create hook for capturing/patching hidden states.

            Args:
                layer_idx: Layer index
                is_clean_pass: Whether this is the clean pass
                component: "block" (whole layer), "attn", or "mlp"
            """
            def hook(module, input, output):
                # Handle tuple output from TransformerBlock (hidden_states, attention_weights)
                if isinstance(output, tuple):
                    hidden_states, attn_weights = output
                else:
                    hidden_states = output
                    attn_weights = None

                # Create unique key for this component
                state_key = (layer_idx, component)

                if is_clean_pass:
                    # Store clean hidden states
                    clean_states[state_key] = hidden_states.detach().clone()
                    # Return output unchanged
                    return output
                else:
                    # Patch states if needed - create a copy to avoid in-place modification issues
                    if any(target_layer == layer_idx for _, target_layer in states_to_patch):
                        hidden_states = hidden_states.clone()
                        for token_idx, target_layer in states_to_patch:
                            if target_layer == layer_idx:
                                # Restore clean state for this token in noisy samples
                                hidden_states[1:, token_idx, :] = clean_states[state_key][0, token_idx, :]

                # Return in the same format as received
                if isinstance(output, tuple):
                    return (hidden_states, attn_weights)
                else:
                    return hidden_states
            return hook

        # Determine which component to hook
        if kind == "mlp":
            component = "mlp"
            hook_target_fn = lambda idx: self.model.blocks[idx].ffn
        elif kind == "attn":
            component = "attn"
            hook_target_fn = lambda idx: self.model.blocks[idx].attn
        else:
            # Hook the whole transformer block
            component = "block"
            hook_target_fn = lambda idx: self.model.blocks[idx]

        # Run clean pass to collect states
        handles = []
        for layer_idx in range(self.num_layers):
            handle = hook_target_fn(layer_idx).register_forward_hook(
                make_hook(layer_idx, is_clean_pass=True, component=component)
            )
            handles.append(handle)

        with torch.no_grad():
            # Clean forward pass (first sample)
            _ = self.model(input_ids)

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Add noise to embeddings and run noisy pass with patching
        handles = []
        for layer_idx in range(self.num_layers):
            handle = hook_target_fn(layer_idx).register_forward_hook(
                make_hook(layer_idx, is_clean_pass=False, component=component)
            )
            handles.append(handle)

        # Hook to corrupt embeddings
        def corrupt_embeddings(module, input, output):
            # Add noise to subject tokens in samples 1:
            s_start, s_end = subject_range
            noise = torch.randn(
                num_samples, s_end - s_start, output.shape[-1],
                device=output.device
            ) * noise_std
            output[1:, s_start:s_end, :] += noise
            return output

        # Corrupt at embedding layer
        embed_handle = None
        if hasattr(self.model, 'token_embed'):
            # Hook after embedding + positional encoding
            def corrupt_after_embed(module, input, output):
                # This hooks the dropout after embeddings
                return corrupt_embeddings(module, input, output)
            embed_handle = self.model.dropout.register_forward_hook(corrupt_after_embed)

        with torch.no_grad():
            outputs = self.model(input_ids_batch)
            logits = outputs["logits"][:, -1, :]  # [batch_size, vocab_size]
            probs = torch.softmax(logits, dim=-1)
            target_probs = probs[1:, target_token_id]  # Exclude clean sample
            avg_prob = target_probs.mean().item()

        # Clean up hooks
        for handle in handles:
            handle.remove()
        if embed_handle is not None:
            embed_handle.remove()

        return avg_prob

    def _get_embedding_std(self) -> float:
        """Compute standard deviation of embedding weights."""
        if hasattr(self.model, 'token_embed'):
            embed_weight = self.model.token_embed.weight
            return embed_weight.std().item()
        return 1.0

    def locate_important_layer(
        self,
        s: str,
        r: str,
        o_target: str,
        noise_level: float = 3.0,
        num_samples: int = 10,
        token_strategy: str = "subject_last"
    ) -> int:
        """
        Locate the most important layer for a fact.

        Args:
            s: Subject
            r: Relation
            o_target: Target object
            noise_level: Noise level in standard deviations
            num_samples: Number of samples
            token_strategy: Which token to focus on ("subject_last" or "all")

        Returns:
            Layer index with highest causal effect
        """
        result = self.trace_important_states(s, r, o_target, noise_level, num_samples)

        if token_strategy == "subject_last":
            # Focus on last subject token
            subject_last_idx = result.subject_range[1] - 1
            layer_scores = result.scores[subject_last_idx, :]
        else:
            # Average over all tokens
            layer_scores = result.scores.mean(dim=0)

        best_layer = torch.argmax(layer_scores).item()
        return best_layer

    def locate_important_layer_with_scores(
        self,
        s: str,
        r: str,
        o_target: str,
        noise_level: float = 3.0,
        num_samples: int = 10,
        token_strategy: str = "subject_last"
    ) -> Tuple[int, List[Dict], torch.Tensor]:
        """
        Locate important layer and return detailed scores.

        Args:
            s: Subject
            r: Relation
            o_target: Target object
            noise_level: Noise level in standard deviations
            num_samples: Number of samples
            token_strategy: Which token to focus on

        Returns:
            Tuple of:
                - best_layer: Layer index with highest effect
                - layer_effects: List of dicts with layer and effect scores
                - token_layer_grid: Full [num_tokens, num_layers] grid of scores
        """
        result = self.trace_important_states(s, r, o_target, noise_level, num_samples)

        if token_strategy == "subject_last":
            # Focus on last subject token
            subject_last_idx = result.subject_range[1] - 1
            layer_scores = result.scores[subject_last_idx, :]
        else:
            # Average over all tokens
            layer_scores = result.scores.mean(dim=0)

        best_layer = torch.argmax(layer_scores).item()

        # Create list of effects
        layer_effects = []
        for layer_idx in range(self.num_layers):
            layer_effects.append({
                'layer': layer_idx,
                'effect': layer_scores[layer_idx].item()
            })

        # Return full grid for visualization
        return best_layer, layer_effects, result.scores