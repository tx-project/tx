"""Tinker server implementation with background task handlers."""

from typing import Any

import jax
from jax import numpy as jnp
from flax import nnx
import optax
from transformers import AutoConfig

from tx.utils.models import get_dtype, get_model_class, load_checkpoint


class TinkerServer:
    """Server class that manages models and training operations."""

    def __init__(self):
        self.models: dict[str, dict[str, Any]] = {}
        self.futures: dict[str, dict[str, Any]] = {}

    def create_model_task(self, model_id: str, request_id: str, base_model: str, lora_config: Any):
        """Background task to load model weights."""
        try:
            # Load config and create actual model
            config = AutoConfig.from_pretrained(base_model)
            model_class = get_model_class(config)

            # Create model with appropriate mesh
            mesh = jax.make_mesh((1, 1), ("dp", "tp"))
            with jax.set_mesh(mesh):
                model = model_class(config, dtype=get_dtype(config.dtype), rngs=nnx.Rngs(0))

            # Load pretrained weights from HuggingFace
            ## load_checkpoint(base_model, config, model)

            # TODO: Apply LoRA configuration if provided
            # if lora_config:
            #     raise NotImplementedError("LoRA configuration not yet implemented")

            # Store model data
            self.models[model_id] = {
                "model_id": model_id,
                "base_model": base_model,
                "lora_config": lora_config,
                "status": "ready",
                "model": model,
                "config": config,
                "gradients": None,  # Will store accumulated gradients
            }

            # Store complete model response for future retrieval
            self.futures[request_id] = {
                "model_id": model_id,
                "base_model": base_model,
                "lora_config": lora_config.model_dump() if lora_config else None,
                "status": "ready",
                "request_id": request_id,
            }
        except Exception as e:
            self.models[model_id]["status"] = "failed"
            self.models[model_id]["error"] = str(e)
            self.futures[request_id] = {"status": "failed", "error": str(e)}

    def forward_backward_task(self, model_id: str, request_id: str, forward_backward_input: dict[str, Any]):
        """Background task to compute gradients."""
        try:
            model_data = self.models[model_id]
            model = model_data["model"]

            # Extract tokens from examples
            data = forward_backward_input["data"]
            input_ids_list = []
            for item in data:
                tokens = [t for chunk in item["model_input"]["chunks"] for t in chunk["tokens"]]
                input_ids_list.append(tokens)

            # Pad sequences to same length
            max_len = max(len(seq) for seq in input_ids_list)
            padded = [seq + [0] * (max_len - len(seq)) for seq in input_ids_list]
            input_ids = jnp.array(padded, dtype=jnp.int32)

            # Define loss function
            def loss_fn(model):
                logits = model(input_ids)["logits"]
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits=logits[:, :-1, :],
                    labels=input_ids[:, 1:]
                )
                return loss.mean()

            # Compute loss and gradients
            loss_value, grads = nnx.value_and_grad(loss_fn)(model)

            # Accumulate gradients
            if model_data["gradients"] is None:
                model_data["gradients"] = grads
            else:
                model_data["gradients"] = jax.tree.map(
                    lambda g1, g2: g1 + g2, model_data["gradients"], grads
                )

            # Store result
            self.futures[request_id] = {
                "loss_fn_output_type": "scalar",
                "loss_fn_outputs": [{
                    "loss": {
                        "data": [float(loss_value)],
                        "dtype": "float32",
                        "shape": [1]
                    }
                }],
                "metrics": {},
                "status": "completed"
            }
        except Exception as e:
            import traceback
            self.futures[request_id] = {
                "status": "failed",
                "error": f"{str(e)}\n{traceback.format_exc()}"
            }

    def optim_step_task(self, model_id: str, request_id: str, adam_params: Any):
        """Background task to update model parameters."""
        try:
            model_data = self.models[model_id]
            model = model_data["model"]

            if model_data["gradients"] is None:
                raise ValueError("No gradients accumulated. Call forward_backward first.")

            # Create or retrieve optimizer
            if "optimizer" not in model_data:
                optimizer = nnx.Optimizer(
                    model,
                    optax.adamw(
                        learning_rate=adam_params.lr,
                        b1=adam_params.betas[0],
                        b2=adam_params.betas[1],
                        eps=adam_params.eps,
                        weight_decay=adam_params.weight_decay
                    ),
                    wrt=nnx.Param
                )
                model_data["optimizer"] = optimizer
            else:
                optimizer = model_data["optimizer"]

            # Apply gradients
            optimizer.update(model, model_data["gradients"])

            # Clear accumulated gradients
            model_data["gradients"] = None

            self.futures[request_id] = {
                "status": "completed"
            }
        except Exception as e:
            self.futures[request_id] = {
                "status": "failed",
                "error": str(e)
            }
