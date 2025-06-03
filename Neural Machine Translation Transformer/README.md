# Neural Machine Translation Transformer

This project implements a complete Transformer model for Neural Machine Translation (NMT) from scratch using PyTorch, following the architecture described in the "Attention Is All You Need" paper.

## Features

- Complete encoder-decoder transformer architecture
- Multi-head self-attention and cross-attention mechanisms
- Positional encoding
- Masking for padding and future tokens
- Translation inference with beam search
- Configurable model dimensions and hyperparameters

## Technical Details

- **Architecture**: 6-layer encoder, 6-layer decoder, 8-head attention
- **Model Dimension**: 512 with 2048 feed-forward dimension
- **Implementation**: Pure PyTorch with no transformer-specific libraries
- **Training**: Supports teacher forcing and label smoothing

## Usage

The implementation can be used for:
- Neural machine translation between any language pair
- Text summarization
- Dialogue generation
- Any sequence-to-sequence task

## Requirements

```
torch>=1.7.0
numpy>=1.19.0
```

## Example

```python
# Initialize model
model = Transformer(
    src_vocab_size=30000,
    tgt_vocab_size=30000,
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048
)

# Create inputs (tokenized sentences)
src_tokens = torch.tensor([[101, 2054, 2003, 2019, 3319, 102]])  # Example: [SOS, "What", "is", "the", "weather", EOS]
tgt_tokens = torch.tensor([[101, 2054, 2003, 2019, 3319, 102]])  # For training

# Create masks
src_mask, tgt_mask = model.create_masks(src_tokens, tgt_tokens)

# Forward pass (training)
output = model(src_tokens, tgt_tokens, src_mask, tgt_mask)

# Translation (inference)
translated = model.translate(src_tokens)
```