from flax import nnx
import jax
from jax import numpy as jnp
from transformers import Qwen3Config


def Param(*shape: int, dtype: jnp.dtype, kernel_init: nnx.Initializer, rngs: nnx.Rngs):
    return nnx.Param(kernel_init(rngs.param(), shape, dtype))


class RMSNorm(nnx.Module):
    def __init__(self, size: int, *, eps: float = 1e-6, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.eps = eps
        self.weight = Param(size, dtype=dtype, kernel_init=nnx.with_partitioning(nnx.initializers.normal(), jax.P(None)), rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return self.weight * x / rms


class MultiHeadProj(nnx.Module):

    def __init__(self, subscripts: str, *shape: int, sharding: jax.P, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.subscripts = subscripts
        self.weight = Param(*shape, dtype=dtype, kernel_init=nnx.with_partitioning(nnx.initializers.normal(), sharding), rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.einsum(self.subscripts, x, self.weight)


def apply_rope(inputs: jax.Array, position_ids: jax.Array, head_dim: int, theta: int) -> jax.Array:
    fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.float32) / head_dim
    timescale = jnp.pow(theta, fraction)
    x = (position_ids[..., None] / timescale[None, None, :])[..., None, :]
    sin, cos = jnp.sin(x), jnp.cos(x)
    a, b = jnp.split(inputs, 2, axis=-1)
    return jnp.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1).astype(inputs.dtype)


class Qwen3Attention(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, 'head_dim', None) or config.hidden_size // self.num_heads
        
        self.q_proj = MultiHeadProj("BMK,KNH->BMNH", config.hidden_size, self.num_heads, self.head_dim, sharding=jax.P(None, "tp", None), dtype=dtype, rngs=rngs)
        self.k_proj = MultiHeadProj("BMK,KNH->BMNH", config.hidden_size, self.num_kv_heads, self.head_dim, sharding=jax.P(None, "tp", None), dtype=dtype, rngs=rngs)
        self.v_proj = MultiHeadProj("BMK,KNH->BMNH", config.hidden_size, self.num_kv_heads, self.head_dim, sharding=jax.P(None, "tp", None), dtype=dtype, rngs=rngs)
        self.o_proj = MultiHeadProj("BMKH,KHN->BMN", self.num_heads, self.head_dim, config.hidden_size, sharding=jax.P("tp", None, None), dtype=dtype, rngs=rngs)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        *,
        attention_mask: jax.Array | None = None,
        output_attentions: bool | None = None
    ) -> tuple[jax.Array, jax.Array | None]:
        q = self.q_norm(self.q_proj(x))
        k = self.k_norm(self.k_proj(x))
        v = self.v_proj(x)

        position_ids = jnp.arange(x.shape[1])[None, :].repeat(x.shape[0], axis=0)

        q = apply_rope(q, position_ids, self.head_dim, self.config.rope_theta)
        k = apply_rope(k, position_ids, self.head_dim, self.config.rope_theta)

        if self.num_kv_heads != self.num_heads:
            num_groups = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, num_groups, axis=2)
            v = jnp.repeat(v, num_groups, axis=2)

        # Use jax.nn.dot_product_attention for efficient, platform-agnostic attention
        # This automatically uses FlashAttention (via CuDNN) on GPU or optimized XLA kernels on TPU
        # q, k, v are already in BTNH format (B=batch, T=seq_len, N=num_heads, H=head_dim)

        # Prepare mask for dot_product_attention if needed
        # attention_mask is [B, T] indicating which tokens are valid (1) vs padding (0)
        mask = None
        if attention_mask is not None:
            # Convert to boolean: [B, T] -> [B, 1, 1, T] for broadcasting
            mask = attention_mask[:, None, None, :].astype(bool)

        attn_output = jax.nn.dot_product_attention(
            q, k, v,
            scale=1.0 / jnp.sqrt(self.head_dim),
            mask=mask,
            is_causal=True,
        )

        # Compute attention weights if requested (using naive implementation for backward compatibility)
        if output_attentions:
            attn_weights = jnp.einsum("BMNH,BTNH->BNMT", q, k) / jnp.sqrt(self.head_dim)
            causal_mask = jnp.tril(jnp.ones((x.shape[1], x.shape[1])))[None, None, :, :]
            if attention_mask is not None:
                causal_mask *= attention_mask[:, None, None, :]
            attn_weights = jnp.where(causal_mask == 0, -jnp.inf, attn_weights)
            attn_weights = nnx.softmax(attn_weights, axis=-1)
        else:
            attn_weights = None

        return self.o_proj(attn_output), attn_weights
        

class Qwen3MLP(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.gate_proj = nnx.Linear(
            config.hidden_size, config.intermediate_size, use_bias=False, dtype=dtype, param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P(None, "tp")), rngs=rngs
        )
        self.up_proj = nnx.Linear(
            config.hidden_size, config.intermediate_size, use_bias=False, dtype=dtype, param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P(None, "tp")), rngs=rngs
        )
        self.down_proj = nnx.Linear(
            config.intermediate_size, config.hidden_size, use_bias=False, dtype=dtype, param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P("tp", None)), rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(nnx.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Experts(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
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

    def __call__(self, hidden_states: jax.Array, router_logits: jax.Array) -> jax.Array:
        # Get top-k experts for each token and compute routing weights
        routing_weights, selected_experts = jax.lax.top_k(
            router_logits, k=self.config.num_experts_per_tok
        )
        routing_weights = nnx.softmax(routing_weights, axis=-1)

        # Prepare for ragged_dot by sorting tokens based on their assigned expert
        selected_experts_flat = selected_experts.ravel()
        hidden_states_expanded = jnp.repeat(hidden_states, self.config.num_experts_per_tok, axis=0)
        sort_indices = jnp.argsort(selected_experts_flat)
        hidden_states_sorted = hidden_states_expanded[sort_indices]
        group_sizes = jnp.bincount(selected_experts_flat, length=self.config.num_experts)

        # Apply expert layers using ragged_dot
        gate_out = jax.lax.ragged_dot(hidden_states_sorted, self.gate_proj.value, group_sizes)
        up_out = jax.lax.ragged_dot(hidden_states_sorted, self.up_proj.value, group_sizes)
        down_out = jax.lax.ragged_dot(
            nnx.silu(gate_out) * up_out, self.down_proj.value, group_sizes
        )

        # Unsort and combine the expert outputs
        unsort_indices = jnp.argsort(sort_indices)
        unsorted_out = down_out[unsort_indices]
        reshaped_out = unsorted_out.reshape(-1, self.config.num_experts_per_tok, self.config.hidden_size)
        return jnp.sum(reshaped_out * routing_weights[..., None], axis=1)


class Qwen3MoeSparseMoeBlock(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.gate = nnx.Linear(
            config.hidden_size, config.num_experts,
            use_bias=False, dtype=dtype, param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P(None, None)), rngs=rngs,
        )
        self.experts = Qwen3Experts(config, dtype=dtype, rngs=rngs)

    def __call__(self, hidden_states: jax.Array, *, return_router_logits: bool = False) -> jax.Array | tuple[jax.Array, jax.Array]:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
        router_logits = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, router_logits)
        hidden_states = hidden_states.reshape(original_shape)

        if return_router_logits:
            return hidden_states, router_logits
        return hidden_states
        

class Qwen3DecoderLayer(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.self_attn = Qwen3Attention(config, dtype=dtype, rngs=rngs)
        if getattr(config, "num_experts", None):
            self.mlp = Qwen3MoeSparseMoeBlock(config, dtype=dtype, rngs=rngs)
        else:
            self.mlp = Qwen3MLP(config, dtype=dtype, rngs=rngs)

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


class Qwen3Model(nnx.Module):

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
        self.layers = nnx.List([Qwen3DecoderLayer(config, dtype=dtype, rngs=rngs) for _ in range(config.num_hidden_layers)])
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


class Qwen3ForCausalLM(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.model = Qwen3Model(config, dtype=dtype, rngs=rngs)
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

