import os
from pathlib import Path

from datasets import IterableDataset, load_dataset
import jax.numpy as jnp
from flax import nnx
import optax
import safetensors.numpy
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig
import typer

from xtrain.models import Qwen3ForCausalLM
from xtrain.utils import get_param_mapping

app = typer.Typer()


def save_checkpoint(config: PretrainedConfig, model: nnx.Module, filename: str | os.PathLike) -> None:
    param_mapping = get_param_mapping(config, model)
    model_params = nnx.to_flat_state(nnx.state(model))
    tensors = {}
    for path, param in model_params:
        if "rngs" in path:
            continue
        key = param_mapping[path]
        if path[-2] in {"q_proj", "k_proj", "v_proj"}:
            param = param.reshape(param.shape[0], -1)
        if path[-2] in {"o_proj"}:
            param = param.reshape(-1, param.shape[-1])
        if path[-2] == "embed_tokens":
            tensors[key] = param
        else:
            tensors[key] = param.T
    safetensors.numpy.save_file(tensors, filename)


# def cross_entropy_loss(logits, targets):
#     assert logits.ndim == targets.ndim + 1, f"Shapes are {logits.shape} for logits and {targets.shape} for targets."
#     onehot_targets = nnx.one_hot(targets, logits.shape[-1])
#     loss = -jnp.sum(onehot_targets * nnx.log_softmax(logits), axis=-1)
#     return loss.sum()


def loss_fn(model, batch):
    logits = model(batch["text"], attention_mask=batch["attention_mask"])["logits"]
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["target"]
    )
    return loss.mean(), logits


@nnx.jit
def train_step(model, optimizer: nnx.Optimizer, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    optimizer.update(model, grads)
    return loss

    
@app.command()
def main(
    dataset: str = typer.Option(..., "--dataset", help="HuggingFace dataset to use for training"),
    output_dir: Path = typer.Option(..., "--output-dir", help="The output directory where the model predictions and checkpoints will be written"),
    per_device_batch_size: int = typer.Option(..., "--per-device-batch-size", help="Batch size per device accelerator for training."),
) -> None:
    train_dataset = load_dataset(dataset, split="train")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    model = Qwen3ForCausalLM(config, rngs=nnx.Rngs(0))

    optimizer = nnx.Optimizer(
        model, optax.adamw(0.002, weight_decay=0.1), wrt=nnx.Param
    )

    for step, data in enumerate(train_dataset.iter(batch_size=per_device_batch_size)):
        batch = {k: jnp.asarray(v) for k, v in tokenizer(data["text"], return_tensors="np", padding=True).items()}
        model.train()
        input_batch = {
            "text": batch["input_ids"][:,:-1],
            "attention_mask": batch["attention_mask"][:,:-1],
            "target": batch["input_ids"][:, 1:],
        }
        loss = train_step(model, optimizer, input_batch)
        print("step", step, "loss", loss)

        if step % 10 == 0:
            save_checkpoint(config, model, output_dir / "model.safetensors")


if __name__ == "__main__":
    app()

