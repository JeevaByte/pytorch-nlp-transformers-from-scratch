# GPT-2 Implementation from Scratch

This project implements the GPT-2 architecture from scratch using PyTorch, demonstrating deep understanding of autoregressive language models and transformer decoder architecture.

## Features

- Complete GPT-2 architecture implementation
- Masked multi-head self-attention mechanism
- Custom layer normalization
- Text generation capabilities with various decoding strategies:
  - Temperature sampling
  - Top-k filtering
  - Nucleus (top-p) sampling
- Weight tying between embedding and output layers

## Technical Details

- **Architecture**: Configurable layers, embedding dimensions, and attention heads
- **Default Configuration**: 12-layer, 768-hidden, 12-heads (~124M parameters)
- **Implementation**: Pure PyTorch with no transformer-specific libraries
- **Training**: Supports causal language modeling

## Usage

The implementation can be used for:
- Text generation
- Fine-tuning on specific domains
- Transfer learning for downstream tasks

## Requirements

```
torch>=1.7.0
numpy>=1.19.0
```

## Example

```python
# Initialize model
model = GPT2LMHeadModel(
    vocab_size=50257,
    n_ctx=1024,
    n_embd=768,
    n_layer=12,
    n_head=12
)

# Generate text
input_ids = torch.tensor([[50, 100, 200, 300]])  # Example token IDs
generated_text = model.generate(
    input_ids,
    max_length=50,
    temperature=0.7,
    top_k=40,
    top_p=0.9
)
```