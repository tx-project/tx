# tx: Cross-Platform Transformer Training

tx (**t**ransformers **x**-platform) is a JAX/OpenXLA-based library
designed for training transformers and other neural networks. Built on
OpenXLA, tx enables you to run the same code seamlessly across diverse
hardware platforms—including GPUs, TPUs, AWS Trainium, and Tenstorrent
accelerators—without the complexity of adapting to platform-specific
APIs or execution models like those found in PyTorch/XLA.

Our philosophy centers on simplicity without sacrificing power. We've
crafted the library to feel intuitive and familiar to developers
coming from the PyTorch and Hugging Face ecosystems, making the
transition to cross-platform training as smooth as possible.

Key Benefits
- Write once, run anywhere: Single codebase that works across all major AI accelerators
- Familiar API: Designed with PyTorch and Hugging Face developers in mind
- Clean and maintainable: Simple, powerful code that doesn't compromise on capability
