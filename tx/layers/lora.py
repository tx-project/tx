from flax import nnx
import jax
from jax import numpy as jnp


def Param(*shape: int, dtype: jnp.dtype, kernel_init: nnx.Initializer, rngs: nnx.Rngs):
    return nnx.Param(kernel_init(rngs.param(), shape, dtype))


class LoRAMixin:
    """A mixin for flax NNX modules to add multi-adapter LoRA support.

    This mixin adds LoRA parameters (lora_A, lora_B) and methods to apply
    the low-rank adaptation to a base module's output. It is designed to
    be used with layers like nnx.Linear.
    """

    lora_scaling: nnx.Param | None
    lora_A: nnx.Param | None
    lora_B: nnx.Param | None

    def init_lora(
        self, *, max_lora_adapters: int, in_features: int, out_features: int,
        max_lora_rank: int, dtype: jnp.dtype, rngs: nnx.Rngs,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.max_lora_adapters = max_lora_adapters
        self.max_lora_rank = max_lora_rank

        if max_lora_adapters == 0:
            self.lora_scaling = None
            self.lora_A = None
            self.lora_B = None
        else:
            self.lora_scaling = Param(
                max_lora_adapters, dtype=dtype,
                kernel_init=nnx.initializers.constant(1.0), rngs=rngs,
            )
            self.lora_A = Param(
                max_lora_adapters, in_features, max_lora_rank, dtype=dtype,
                kernel_init=nnx.initializers.normal(stddev=0.02), rngs=rngs,
            )
            self.lora_B = Param(
                max_lora_adapters, max_lora_rank, out_features, dtype=dtype,
                kernel_init=nnx.initializers.zeros_init(), rngs=rngs,
            )

    def apply_lora(
        self,
        x: jax.Array,
        base_output: jax.Array,
        adapter_indices: jax.Array | None,
    ) -> jax.Array:
        if self.max_lora_adapters == 0:
            return base_output

        batch_size = x.shape[0]
        assert adapter_indices is not None, "If max_lora_adapters > 0, adapter_indices need to be specified"
        assert adapter_indices.shape[0] == batch_size

        x_flat = x.reshape(batch_size, -1, self.in_features)
        A = self.lora_A.value[adapter_indices]
        B = self.lora_B.value[adapter_indices]
        scaling = self.lora_scaling.value[adapter_indices]

        lora_output = jnp.einsum('bsi,bir,bro->bso', x_flat, A, B)
        lora_output = lora_output.reshape(base_output.shape) * scaling[:, None, None]
        return base_output + lora_output


class LoRALinear(LoRAMixin, nnx.Linear):
    """An nnx.Linear layer with multi-adapter LoRA support."""

    def __init__(
        self, in_features: int, out_features: int, *,
        max_lora_adapters: int = 0, max_lora_rank: int = 8,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype | None = None,
        use_bias: bool = True,
        kernel_init: nnx.Initializer | None = None,
        bias_init: nnx.Initializer | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        param_dtype = param_dtype or dtype
        if kernel_init is None:
            kernel_init = nnx.initializers.lecun_normal()
        if use_bias and bias_init is None:
            bias_init = nnx.initializers.zeros_init()

        super().__init__(in_features, out_features, use_bias=use_bias, dtype=dtype, param_dtype=param_dtype,
            kernel_init=kernel_init, bias_init=bias_init, rngs=rngs,
        )
        self.init_lora(in_features=in_features, out_features=out_features, max_lora_adapters=max_lora_adapters, max_lora_rank=max_lora_rank,
            dtype=param_dtype, rngs=rngs,
        )

    def __call__(self, x: jax.Array, adapter_indices: jax.Array | None = None) -> jax.Array:
        base_out = super().__call__(x)
        return self.apply_lora(x, base_out, adapter_indices)
