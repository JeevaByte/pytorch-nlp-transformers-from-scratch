# RoBERTa Implementation from Scratch

This project implements the RoBERTa (Robustly Optimized BERT Pretraining Approach) architecture from scratch using PyTorch, following the design described in the "RoBERTa: A Robustly Optimized BERT Pretraining Approach" paper.

## Features

- Complete RoBERTa architecture implementation
- Multi-head self-attention mechanism
- Layer normalization and residual connections
- Multiple task-specific heads:
  - Sequence classification
  - Token classification (NER)
  - Question answering
- Dynamic attention masking

## Technical Details

- **Architecture**: 12-layer, 768-hidden, 12-heads (~125M parameters)
- **Implementation**: Pure PyTorch with no transformer-specific libraries
- **Training**: Supports masked language modeling with dynamic masking
- **Improvements over BERT**: 
  - Removes next-sentence prediction
  - Uses dynamic masking
  - Larger batch sizes
  - Longer training

## Usage

The implementation can be used for:
- Fine-tuning on downstream tasks
- Text classification
- Named entity recognition
- Question answering
- Any BERT-compatible task

## Requirements

```
torch>=1.7.0
numpy>=1.19.0
```

## Example

```python
# Initialize model for sequence classification
model = RobertaForSequenceClassification(
    vocab_size=50265,
    num_labels=2,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12
)

# Create inputs (tokenized sentences)
input_ids = torch.tensor([[0, 1000, 2000, 3000, 2]])  # Example token IDs with [CLS] and [SEP]
attention_mask = torch.tensor([[1, 1, 1, 1, 1]])  # All tokens are real (not padding)

# Forward pass
logits = model(input_ids, attention_mask=attention_mask)
predictions = torch.argmax(logits, dim=1)
```