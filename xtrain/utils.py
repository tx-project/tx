from flax import nnx
from transformers import PretrainedConfig


def get_param_mapping(config: PretrainedConfig, model: nnx.Module) -> dict[tuple, str]:
    "Get the mapping from model parameter paths to safetensors keys."

    def get_key(path: tuple) -> str:
        if path[-1] in {"embedding", "kernel"}:
            path = (*path[:-1], "weight")
        return ".".join(map(str, path))

    param_mapping = {}
    model_params = nnx.to_flat_state(nnx.state(model))
    for path, _ in model_params:
        key = get_key(path)
        if path[-2] == "lm_head" and config.tie_word_embeddings:
            key = "model.embed_tokens.weight"
        param_mapping[path] = key
    return param_mapping
