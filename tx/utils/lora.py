"""Utilities for creating LoRA shadow models that share base weights."""

from flax import nnx
import jax
from jax import numpy as jnp


class ScaledLoRA(nnx.Module):
    """LoRA wrapper with proper alpha scaling."""

    def __init__(self, in_features: int, lora_rank: int, out_features: int,
                 lora_alpha: float, base_module: nnx.Module,
                 dtype: jnp.dtype = jnp.float32, *, rngs: nnx.Rngs):
        self.base_module = base_module
        self.lora_a = nnx.Param(nnx.initializers.normal(stddev=1.0)(rngs.param(), (in_features, lora_rank), dtype))
        self.lora_b = nnx.Param(nnx.initializers.zeros_init()(rngs.param(), (lora_rank, out_features), dtype))
        self.scaling = lora_alpha / lora_rank
        self.dtype = dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        # Apply LoRA with scaling: base(x) + scaling * (x @ A @ B)
        base_out = self.base_module(x)
        lora_out = x @ self.lora_a.value @ self.lora_b.value
        return base_out + self.scaling * lora_out


def create_lora_shadow_model(base_model: nnx.Module, lora_rank: int, lora_alpha: float = 16.0,
                              target_modules: list[str] | None = None, dtype: jnp.dtype = jnp.float32, *,
                              rngs: nnx.Rngs) -> nnx.Module:
    """
    Create a shadow model with LoRA adapters on top of a base model.

    The shadow model shares the base weights with the original model and only adds
    LoRA parameters (A and B matrices) that can be different for each forward pass.

    Args:
        base_model: The base model to wrap with LoRA
        lora_rank: Rank of the LoRA adaptation
        lora_alpha: LoRA scaling parameter
        target_modules: List of module name patterns to wrap (e.g., ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])
                       If None, wraps all Linear layers
        dtype: Data type for LoRA parameters
        rngs: Random number generator state

    Returns:
        A new model with LoRA adapters wrapping the specified Linear layers
    """
    # Create a graphdef and state split
    graphdef, state = nnx.split(base_model)
    shadow_model = nnx.merge(graphdef, state)

    # Walk through the model tree and wrap Linear layers with LoRA
    def should_wrap(path: str) -> bool:
        """Check if this module path should be wrapped with LoRA."""
        if target_modules is None:
            return True
        return any(target in path for target in target_modules)

    def wrap_with_lora(module: nnx.Module, path: str = "") -> nnx.Module:
        match module:
            case nnx.Linear() if should_wrap(path):
                return ScaledLoRA(
                    module.in_features,
                    lora_rank,
                    module.out_features,
                    lora_alpha=lora_alpha,
                    base_module=module,
                    dtype=dtype,
                    rngs=rngs
                )
            case nnx.List():
                for i, item in enumerate(module):
                    new_path = f"{path}[{i}]" if path else f"[{i}]"
                    wrap_with_lora(item, new_path)
                return module
            case nnx.Module():
                for name in dir(module):
                    if name.startswith('_'):
                        continue
                    attr = getattr(module, name, None)
                    if isinstance(attr, nnx.Module):
                        new_path = f"{path}.{name}" if path else name
                        wrapped = wrap_with_lora(attr, new_path)
                        setattr(module, name, wrapped)
                return module
            case _:
                return module

    return wrap_with_lora(shadow_model)
