"""Utilities for creating LoRA shadow models that share base weights."""

from flax import nnx
import jax
from jax import numpy as jnp


def create_lora_shadow_model(base_model: nnx.Module, lora_rank: int, lora_alpha: float = 16.0,
                              target_modules: list[str] = None, dtype: jnp.dtype = jnp.float32,
                              rngs: nnx.Rngs = None) -> nnx.Module:
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
    if rngs is None:
        rngs = nnx.Rngs(0)

    # Create a graphdef and state split
    graphdef, state = nnx.split(base_model)
    shadow_model = nnx.merge(graphdef, state)

    # Walk through the model tree and wrap Linear layers with LoRA
    def should_wrap(path: str) -> bool:
        """Check if this module path should be wrapped with LoRA."""
        if target_modules is None:
            return True
        return any(target in path for target in target_modules)

    def wrap_with_lora(module, path=""):
        if isinstance(module, nnx.Linear) and should_wrap(path):
            # Create a LoRA wrapper that references the base Linear layer
            return nnx.LoRA(
                module.in_features,
                lora_rank,
                module.out_features,
                base_module=module,
                dtype=dtype,
                rngs=rngs
            )
        elif isinstance(module, nnx.List):
            # Handle nnx.List specially (e.g., model.layers)
            for i, item in enumerate(module):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                wrap_with_lora(item, new_path)
        elif isinstance(module, nnx.Module):
            # Recursively process submodules
            for name in dir(module):
                if name.startswith('_'):
                    continue
                attr = getattr(module, name, None)
                if attr is None:
                    continue
                if isinstance(attr, (nnx.Linear, nnx.List, nnx.Module)):
                    new_path = f"{path}.{name}" if path else name
                    wrapped = wrap_with_lora(attr, new_path)
                    if wrapped is not attr:
                        setattr(module, name, wrapped)
        return module

    wrap_with_lora(shadow_model)
    return shadow_model
