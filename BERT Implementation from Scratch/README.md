# Custom BERT Implementation from Scratch

This project demonstrates a complete implementation of BERT (Bidirectional Encoder Representations from Transformers) using PyTorch, built entirely from scratch without relying on pre-built transformer libraries.

## Features

- Complete transformer architecture implementation
- Multi-head self-attention mechanism
- Positional embeddings
- Token and segment embeddings
- Layer normalization and residual connections
- Sequence classification capability

## Technical Details

- **Architecture**: 12-layer, 768-hidden, 12-heads
- **Parameters**: ~110M parameters (base model)
- **Implementation**: Pure PyTorch with no transformer-specific libraries
- **Training**: Supports masked language modeling and next sentence prediction

## Usage

The implementation can be used for:
- Fine-tuning on downstream tasks
- Text classification
- Token classification (NER)
- Question answering

## Requirements

```
torch>=1.7.0
numpy>=1.19.0
```

## Example

```python
# Initialize model
model = BERTForSequenceClassification(
    vocab_size=30000,
    num_classes=2,
    d_model=768,
    num_layers=12,
    num_heads=12
)

# Forward pass with dummy data
input_ids = torch.randint(0, 30000, (8, 128))
outputs = model(input_ids)
```