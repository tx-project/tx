import os
import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tx.models import Qwen3ForCausalLM
from tx.models.qwen3 import Qwen3MoeSparseMoeBlock
from tx.utils.models import load_checkpoint


@pytest.mark.parametrize("tp", [1, 2])
def test_qwen3(tp: int):
    if not jax._src.xla_bridge.backends_are_initialized():  # ty: ignore
        jax.config.update("jax_num_cpu_devices", 2)

    if tp > 1 and os.getenv("CI"):
        pytest.skip("TP > 1 currently runs out of memory in the CI")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", attn_implementation="eager", use_safetensors=True)

    inputs = ["The capital of France is", "The most popular programming language is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)
    with torch.no_grad():
        hf_outputs = hf_model(batch.input_ids, attention_mask=batch.attention_mask, output_hidden_states=True, return_dict=True)

    # Save the HF model checkpoint so we can load our model from it
    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)

        config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
        mesh = jax.make_mesh((1, tp), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_checkpoint(tmp, config, model)
        
        outputs = model(batch.input_ids.numpy(), attention_mask=batch.attention_mask.numpy(), output_hidden_states=True)
        assert np.allclose(hf_outputs.hidden_states[0], outputs["hidden_states"][0], rtol=1e-6)
        assert np.allclose(hf_outputs.hidden_states[1], outputs["hidden_states"][1], rtol=1e-3, atol=1e-3)
        assert np.allclose(hf_outputs.hidden_states[-1], outputs["hidden_states"][-1], rtol=1e-3, atol=1e-3)


def test_qwen3_moe_layer():
    model_name = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)
    config = AutoConfig.from_pretrained(model_name)

    hf_moe_layer = hf_model.model.layers[0].mlp
    x = torch.randn(4, 2, config.hidden_size)
    with torch.no_grad():
        hf_final_hidden_states, hf_router_logits = hf_moe_layer.forward(x)

    mesh = jax.make_mesh((1, 1), ("dp", "tp"))
    with jax.set_mesh(mesh):
        moe_layer = Qwen3MoeSparseMoeBlock(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        moe_layer.gate.kernel[:] = hf_moe_layer.gate.weight[:].detach().numpy().T
        for i, expert in enumerate(hf_moe_layer.experts):
            moe_layer.experts.gate_proj[i,:,:] = expert.gate_proj.weight.detach().numpy().T
            moe_layer.experts.up_proj[i,:,:] = expert.up_proj.weight.detach().numpy().T
            moe_layer.experts.down_proj[i,:,:] = expert.down_proj.weight.detach().numpy().T

    final_hidden_states, router_logits = moe_layer(x.numpy(), return_router_logits=True)

    assert np.allclose(hf_router_logits, router_logits, rtol=1e-4)
    assert np.allclose(hf_final_hidden_states, final_hidden_states, rtol=1e-2, atol=1e-2)


def test_qwen3_lora():
    """Test LoRA implementation by comparing with HuggingFace PEFT model."""
    from tx.utils.lora import create_lora_shadow_model

    base_model_name = "Qwen/Qwen3-0.6B"
    lora_adapter = "charent/self_cognition_Alice"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    inputs = ["The capital of France is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    with tempfile.TemporaryDirectory() as base_tmp:
        # Load and save base model
        base_hf_model = AutoModelForCausalLM.from_pretrained(base_model_name, attn_implementation="eager", use_safetensors=True)
        base_hf_model.save_pretrained(base_tmp, safe_serialization=True)

        config = AutoConfig.from_pretrained(base_model_name)

        mesh = jax.make_mesh((1, 1), ("dp", "tp"))
        with jax.set_mesh(mesh):
            # Create base model and load base weights
            base_model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
            load_checkpoint(base_tmp, config, base_model)


            # Load LoRA config first to get the rank and alpha
            from peft import LoraConfig, get_peft_model

            # Get the original config but override target_modules to only MLP
            original_lora_config = LoraConfig.from_pretrained(lora_adapter)

            # Create a new config with only MLP target modules for testing
            mlp_only_config = LoraConfig(
                r=original_lora_config.r,
                lora_alpha=original_lora_config.lora_alpha,
                target_modules=['gate_proj', 'up_proj', 'down_proj'],
                lora_dropout=original_lora_config.lora_dropout,
                bias=original_lora_config.bias,
                task_type=original_lora_config.task_type,
            )

            # Apply this config to create a new PEFT model with MLP-only LoRA
            hf_lora_model = get_peft_model(base_hf_model, mlp_only_config)
            hf_lora_model.eval()

            # Load the adapter weights from the original adapter
            hf_lora_model.load_adapter(lora_adapter, adapter_name='default')
            lora_config = mlp_only_config

            # Create shadow model with LoRA adapters on MLP layers only, using the correct config
            model = create_lora_shadow_model(
                base_model,
                lora_rank=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                target_modules=["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
                dtype=jnp.float32,
                rngs=nnx.Rngs(1)
            )

            # Get outputs from LoRA model for comparison
            with torch.no_grad():
                hf_outputs = hf_lora_model(batch.input_ids, attention_mask=batch.attention_mask, output_hidden_states=True, return_dict=True)

            # Load LoRA adapter weights from the PEFT model
            for i, layer in enumerate(model.model.layers):
                if hasattr(layer.mlp, 'gate_proj') and hasattr(layer.mlp.gate_proj, 'lora_a'):
                    hf_layer = hf_lora_model.base_model.model.model.layers[i].mlp

                    # Load gate_proj LoRA weights
                    layer.mlp.gate_proj.lora_a.value = jnp.array(hf_layer.gate_proj.lora_A['default'].weight.detach().numpy().T)
                    layer.mlp.gate_proj.lora_b.value = jnp.array(hf_layer.gate_proj.lora_B['default'].weight.detach().numpy().T)

                    # Load up_proj LoRA weights
                    layer.mlp.up_proj.lora_a.value = jnp.array(hf_layer.up_proj.lora_A['default'].weight.detach().numpy().T)
                    layer.mlp.up_proj.lora_b.value = jnp.array(hf_layer.up_proj.lora_B['default'].weight.detach().numpy().T)

                    # Load down_proj LoRA weights
                    layer.mlp.down_proj.lora_a.value = jnp.array(hf_layer.down_proj.lora_A['default'].weight.detach().numpy().T)
                    layer.mlp.down_proj.lora_b.value = jnp.array(hf_layer.down_proj.lora_B['default'].weight.detach().numpy().T)

        outputs = model(batch.input_ids.numpy(), attention_mask=batch.attention_mask.numpy(), output_hidden_states=True)
        assert np.allclose(hf_outputs.logits, outputs["logits"], rtol=1e-3, atol=1e-3)
