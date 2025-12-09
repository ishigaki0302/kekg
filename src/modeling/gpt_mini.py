"""GPT-2 compatible mini language model for SRO learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GPTConfig:
    """GPT model configuration."""
    vocab_size: int = 10000
    n_layers: int = 12
    n_heads: int = 8
    d_model: int = 512
    d_mlp: int = 2048  # 4 * d_model typically
    max_seq_len: int = 8  # For SRO, we only need 3 tokens
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads

        # Q, K, V projections
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            return_attention: Whether to return attention weights

        Returns:
            Output tensor [batch, seq_len, d_model]
            Optional attention weights [batch, n_heads, seq_len, seq_len]
        """
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=-1)

        # Reshape to [batch, n_heads, seq_len, d_head]
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.d_head ** 0.5))

        # Apply causal mask
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = attn_weights @ v  # [batch, n_heads, seq_len, d_head]

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)

        if return_attention:
            return out, attn_weights
        return out, None


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_mlp)
        self.w2 = nn.Linear(config.d_mlp, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.w1(x)
        x = F.gelu(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and FFN."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with residual connections."""
        # Attention block
        attn_out, attn_weights = self.attn(self.ln1(x), return_attention=return_attention)
        x = x + attn_out

        # FFN block
        x = x + self.ffn(self.ln2(x))

        return x, attn_weights


class GPTMini(nn.Module):
    """Mini GPT model for SRO knowledge learning."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)

        # Position embeddings
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embeddings (weight sharing)
        self.lm_head.weight = self.token_embed.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden_states: bool = False,
        return_attention: bool = False
    ) -> dict:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            return_hidden_states: Whether to return all layer hidden states
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with:
                - logits: [batch, seq_len, vocab_size]
                - hidden_states: list of [batch, seq_len, d_model] (optional)
                - attentions: list of attention weights (optional)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Embeddings
        token_emb = self.token_embed(input_ids)  # [B, T, d_model]
        pos_ids = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
        pos_emb = self.pos_embed(pos_ids)  # [1, T, d_model]

        x = self.dropout(token_emb + pos_emb)

        # Collect hidden states and attentions if requested
        hidden_states = [x] if return_hidden_states else None
        attentions = [] if return_attention else None

        # Transformer blocks
        for block in self.blocks:
            x, attn = block(x, return_attention=return_attention)

            if return_hidden_states:
                hidden_states.append(x)
            if return_attention and attn is not None:
                attentions.append(attn)

        # Final layer norm
        x = self.ln_f(x)

        # Output logits
        logits = self.lm_head(x)  # [B, T, vocab_size]

        result = {"logits": logits}
        if return_hidden_states:
            result["hidden_states"] = hidden_states
        if return_attention:
            result["attentions"] = attentions

        return result

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Initial token IDs [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (if None, use greedy)

        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            input_ids_crop = input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            outputs = self.forward(input_ids_crop)
            logits = outputs["logits"]

            # Get logits for last token
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
