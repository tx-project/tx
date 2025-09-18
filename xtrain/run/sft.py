import os

from datasets import IterableDataset, load_dataset
import jax.numpy as jnp
from flax import nnx
import optax
import safetensors.numpy
from transformers import AutoConfig, AutoTokenizer
import typer

from xtrain.models import Qwen3ForCausalLM

app = typer.Typer()


def save_checkpoint(state: nnx.State, filename: str | os.PathLike) -> None:
    params = nnx.to_flat_state(state)
    tensor_dict = {".".join(k): v for k, v in params if "rngs" not in k}
    safetensors.numpy.save_file(tensor_dict, filename)


# def cross_entropy_loss(logits, targets):
#     assert logits.ndim == targets.ndim + 1, f"Shapes are {logits.shape} for logits and {targets.shape} for targets."
#     onehot_targets = nnx.one_hot(targets, logits.shape[-1])
#     loss = -jnp.sum(onehot_targets * nnx.log_softmax(logits), axis=-1)
#     return loss.sum()


def loss_fn(model, batch):
    logits = model(batch['text'][None,:])["logits"]
    print("BBB logits", logits)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['target'][None,:]
    )
    return loss.mean(), logits


# @nnx.jit
def train_step(model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    # metrics.update(loss=loss, logits=logits, labels=batch['label'])
    optimizer.update(model, grads)

    
@app.command()
def main(
    dataset: str = typer.Option(..., "--dataset", help="HuggingFace dataset to use for training.")
) -> None:
    train_dataset = load_dataset(dataset, split="train")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    model = Qwen3ForCausalLM(config, rngs=nnx.Rngs(0))

    optimizer = nnx.Optimizer(
        model, optax.adamw(0.005, 0.9), wrt=nnx.Param
    )
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    ds: IterableDataset = train_dataset.with_format("numpy") # ty: ignore
    for step, data in enumerate(ds.iter(batch_size=1)):
        tokens = jnp.asarray(tokenizer(data["Text"][0])["input_ids"])
        model.train()
        input_batch = {
            "text": tokens[:-1],
            "target": tokens[1:],
        }
        train_step(model, optimizer, metrics, input_batch)

    for metric, value in metrics.compute().items():
        print(metric, value)

    save_checkpoint(nnx.state(model), "checkpoint.safetensors")


if __name__ == "__main__":
    app()

