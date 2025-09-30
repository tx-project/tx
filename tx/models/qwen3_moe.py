from flax import nnx
import jax
from jax import numpy as jnp
from transformers import Qwen3Config

from tx.models.qwen3 import Param, RMSNorm, Qwen3Attention


class Qwen3MoeSparseMoeBlock(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config

        self.gate = nnx.Linear(
            config.hidden_size,
            config.num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P(None, "tp")),
            rngs=rngs,
        )

        self.gate_proj = Param(
            config.num_experts, config.hidden_size, config.moe_intermediate_size,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P(None, None, "tp")),
            rngs=rngs
        )
        self.up_proj = Param(
            config.num_experts, config.hidden_size, config.moe_intermediate_size,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P(None, None, "tp")),
            rngs=rngs
        )
        self.down_proj = Param(
            config.num_experts, config.moe_intermediate_size, config.hidden_size,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P(None, "tp", None)),
            rngs=rngs
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        x_reshaped = x.reshape(-1, self.config.hidden_size)

        # Select top-k experts for each token
        router_logits = self.gate(x_reshaped)
        routing_weights, selected_experts = jax.lax.top_k(router_logits, k=self.config.num_experts_per_tok)
        routing_weights = nnx.softmax(routing_weights, axis=-1)

        # Prepare for ragged_dot by sorting tokens based on their assigned expert
        selected_experts_flat = jnp.ravel(selected_experts)
        x_expanded = jnp.repeat(x_reshaped, self.config.num_experts_per_tok, axis=0)
        sort_indices = jnp.argsort(selected_experts_flat)
        unsort_indices = jnp.argsort(sort_indices)
        x_sorted = x_expanded[sort_indices]
        group_sizes = jnp.bincount(selected_experts_flat, length=self.config.num_experts)

        # Apply expert MLPs using ragged_dot
        gate_out = jax.lax.ragged_dot(x_sorted, self.gate_proj.value, group_sizes)
        up_out = jax.lax.ragged_dot(x_sorted, self.up_proj.value, group_sizes)
        activated = nnx.silu(gate_out) * up_out
        down_out = jax.lax.ragged_dot(activated, self.down_proj.value, group_sizes)

        # Unsort and combine the expert outputs
        unsorted_out = down_out[unsort_indices]
        reshaped_out = unsorted_out.reshape(-1, self.config.num_experts_per_tok, self.config.hidden_size)
        weighted_sum_out = (reshaped_out * routing_weights[..., None]).sum(axis=1)

        return weighted_sum_out.reshape(x.shape), router_logits


class Qwen3MoeDecoderLayer(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.self_attn = Qwen3Attention(config, dtype=dtype, rngs=rngs)
        self.mlp = Qwen3MoeSparseMoeBlock(config, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        *,
        attention_mask: jax.Array | None = None,
        output_attentions: bool | None = None
    ) -> tuple[jax.Array, jax.Array | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, self_attn_weights


class Qwen3MoeModel(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.embed_tokens = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(nnx.initializers.normal(), jax.P("tp", None)),
            rngs=rngs,
        )
        self.layers = nnx.List([Qwen3MoeDecoderLayer(config, dtype=dtype, rngs=rngs) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        input_ids: jax.Array,
        *,
        attention_mask: jax.Array | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None
    ) -> dict[str, jax.Array | list[jax.Array]]:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )

        hidden_states = self.embed_tokens(input_ids)

        all_hidden_states: list[jax.Array] = []
        all_self_attns: list[jax.Array] = []

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states, self_attns = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_self_attns.append(self_attns)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
        }


class Qwen3MoeForCausalLM(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.model = Qwen3MoeModel(config, dtype=dtype, rngs=rngs)
        if not self.config.tie_word_embeddings:
            self.lm_head = nnx.Linear(
                config.hidden_size, config.vocab_size, use_bias=False, dtype=dtype, param_dtype=dtype,
                kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P(None, "tp")), rngs=rngs
            )

    def __call__(
        self,
        input_ids: jax.Array,
        *,
        attention_mask: jax.Array | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None
    ) -> dict[str, jax.Array | list[jax.Array]]:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        hidden_states = outputs["last_hidden_state"]
        if self.config.tie_word_embeddings:
            logits = hidden_states @ self.model.embed_tokens.embedding.value.T
        else:
            logits = self.lm_head(hidden_states)

        return {"logits": logits, **outputs}
