# tx: Cross-platform neural network training

tx (*t*ransformers, *x*-platform) is a JAX/XLA based library for training transformers.
This means the same code can run on a variety of different platforms that OpenXLA
supports like GPUs, TPUs, Trainium, Tenstorrent, etc.

We try to keep the code simple but powerful and write it in a way that is as familiar
to the pytorch and huggingface ecosystem as possible.
