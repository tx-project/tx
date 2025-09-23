import os
from pathlib import Path
import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tx.models import Qwen3ForCausalLM
from tx.utils.models import load_checkpoint


@pytest.mark.parametrize("tp", [1, 2])
def test_qwen3(tp: int):
    if not jax._src.xla_bridge.backends_are_initialized():
        jax.config.update("jax_num_cpu_devices", 2)

    if tp > 1 and os.getenv("CI"):
        pytest.skip("TP > 1 currently runs out of memory in the CI")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", attn_implementation="eager", use_safetensors=True)

    inputs = ["The capital of France is", "The most popular programming language is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)
    with torch.no_grad():
        hf_outputs = hf_model(batch.input_ids, attention_mask=batch.attention_mask, output_hidden_states=True, output_attentions=True, return_dict=True)

    # Save the HF model checkpoint so we can load our model from it
    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)

        config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
        auto_mesh = jax.make_mesh((1, tp), ("dp", "tp"))
        with jax.set_mesh(auto_mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_checkpoint(Path(tmp) / "model.safetensors", config, model)
        
        outputs = model(batch.input_ids.numpy(), attention_mask=batch.attention_mask.numpy(), output_hidden_states=True, output_attentions=True)
        assert np.allclose(hf_outputs.hidden_states[0], outputs["hidden_states"][0], rtol=1e-6)
        assert np.allclose(hf_outputs.attentions[0], outputs["attentions"][0], rtol=1e-4)
        assert np.allclose(hf_outputs.hidden_states[1], outputs["hidden_states"][1], rtol=1e-3, atol=1e-3)
        assert np.allclose(hf_outputs.hidden_states[-1], outputs["hidden_states"][-1], rtol=1e-3, atol=1e-3)
