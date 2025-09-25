import jax.numpy as jnp
from transformers import AutoTokenizer, IterableDataset, PretrainedConfig

from tx.loaders import LoaderIterator


def text(config: PretrainedConfig, dataset: IterableDataset, batch_size: int) -> LoaderIterator:
    "Data loader for text data. It returns an iterator over (batch, metrics) elements."

    tokenizer = AutoTokenizer.from_pretrained(config.name_or_path)
    for data in dataset.iter(batch_size=batch_size):
        # We pad to multiples of 128 here so jax needs to compile less different shapes
        batch = tokenizer(data["text"], return_tensors="np", padding=True, pad_to_multiple_of=128)
        batch = {k: jnp.asarray(v) for k, v in batch.items()}
        yield {
            "text": batch["input_ids"][:,:-1],
            "attention_mask": batch["attention_mask"][:,:-1],
            "target": batch["input_ids"][:, 1:],
        }, {
            "shape": batch["input_ids"].shape,
            "tokens": batch["attention_mask"].sum()
        }
