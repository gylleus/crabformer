# Crabformer 🦀

A transformer-based language model implemented from scratch in Rust, featuring a complete training pipeline with custom backpropagation using an AdamW optimizer.

https://github.com/user-attachments/assets/cba551f0-e2e2-4f6d-888c-79504fed8657

## Overview

I was feeling a bit rusty (heh) on the attention mechanism that transformers utilize for great results within NLP and other fields. While I had read the theory multiple times it didn't quite stick, so I figured that the best way to learn was to just build it from scratch.

Crabformer is a from scratch GPT-style decoder-only transformer model built entirely in Rust without relying on any ML frameworks. It simply uses ndarray for matrices and matrix multiplications, along with rayon for slight boosts in performance using parallel processing.

## Disclaimers

This repo is purely for educational purposes and is not intended for real world use cases. As it only runs on CPU the execution speed is quite low and limits it to small model sizes.

The training pipeline also does not have any test step for calculating the final performance. This is intentionally skipped as the model is a PoC without any concern of overfitting on our small data.

If you are interested in building an actually useful model in Rust you should check out other great projects like [burn](https://github.com/tracel-ai/burn) or [candle](https://github.com/huggingface/candle).

## Quickstart

It is highly recommended to use the `--release` flag to improve execution performance.

### Training

```
cargo run --release -- train
```

### Chat (using pre-trained model)

The repo contains these simple models that were trained on the data in `data/`:
* `chat.ron` - Trained on `dialogs.txt` containing chat data ([source](https://www.kaggle.com/datasets/grafstor/simple-dialogs-for-chatbot))
* `lovecraft.ron` - Trained on several novels from H. P. Lovecraft

```
cargo run --release -- chat -c examples/lovecraft.ron
```

## 🦑 Features

- Multi-head self-attention with causal masking
- Position embeddings
- Layer normalization (pre-norm configuration)
- Feed-forward networks with GELU activation
- Residual connections
- Dropout regularization

Check out [model_architecture.md](./model_architecture.md) for a detailed graph of the model architecture.

### Training Pipeline
- Intelligent data loading with ring buffer shuffling
- Cross-entropy loss
- AdamW optimization with weight decay
- Checkpoint saving/loading (RON format)
- Real-time training metrics dashboard using [Ratatui](https://github.com/ratatui-org/ratatui)

### Text Generation
- Interactive chat interface
- Temperature-controlled sampling
- Top-k sampling that defaults to 1 *(greedy sampling)*
- Configurable generation length


## Future improvements

* GPU support for training/inference
* WASM support to run in browser
* Implement parallelization for more layers
  - Currently this is only done for the linear and attention layers where most compute time is spent
