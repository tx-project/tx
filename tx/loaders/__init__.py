from typing import Iterator

import jax

from tx.loaders.chat import chat
from tx.loaders.text import text


LoaderIterator = Iterator[tuple[dict[str, jax.Array], dict[str, str]]]


__all__ = [
    "chat",
    "text"
]
