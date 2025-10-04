from flax import nnx
import jax
from jax import numpy as jnp


def Param(*shape: int, dtype: jnp.dtype, kernel_init: nnx.Initializer, rngs: nnx.Rngs):
    return nnx.Param(kernel_init(rngs.param(), shape, dtype))


class MultiLoRA(nnx.Module):
    """
    Applies different LoRA adapters to different datapoints in a batch.

    Wraps a base module and adds per-adapter low-rank updates. Each batch element
    can use a different adapter.

    Args:
        base_module: Module to wrap (e.g., nnx.Linear)
        num_adapters: Number of different LoRA adapters
        rank: Rank of the low-rank decomposition
        alpha: LoRA scaling parameter (scales by alpha/rank)
        dtype: Data type
        rngs: Random number generators

    Example:
        base = nnx.Linear(512, 512, use_bias=False, rngs=rngs)
        lora = MultiLoRA(base, num_adapters=8, rank=16, rngs=rngs)

        x = jnp.ones((4, 128, 512))
        adapter_indices = jnp.array([0, 1, 0, 2])
        output = lora(x, adapter_indices)
    """

    def __init__(
        self,
        base_module: nnx.Module,
        num_adapters: int,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        *,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ) -> None:
        self.base = base_module
        self.num_adapters = num_adapters
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # LoRA A: [num_adapters, in_features, rank]
        self.lora_A = Param(
            num_adapters, in_features, rank,
            dtype=dtype,
            kernel_init=nnx.initializers.normal(stddev=0.02),
            rngs=rngs,
        )

        # LoRA B: [num_adapters, rank, out_features] - init to zero
        self.lora_B = Param(
            num_adapters, rank, out_features,
            dtype=dtype,
            kernel_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        adapter_indices: jax.Array,
    ) -> jax.Array:
        """
        Args:
            x: [batch, ..., in_features]
            adapter_indices: [batch] which adapter per batch element (default: all 0)

        Returns:
            [batch, ..., out_features]
        """
        batch_size = x.shape[0]
        base_output = self.base(x)

        # Flatten for computation: [batch, seq, in_features]
        x_flat = x.reshape(batch_size, -1, self.in_features)

        # Select adapters: [batch, in_features, rank] and [batch, rank, out_features]
        A = self.lora_A.value[adapter_indices]
        B = self.lora_B.value[adapter_indices]

        # Compute: x @ A @ B
        lora_output = jnp.einsum('bsi,bir,bro->bso', x_flat, A, B)
        lora_output = lora_output.reshape(base_output.shape) * self.scaling

        return base_output + lora_output
