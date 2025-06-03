# T5 (Text-to-Text Transfer Transformer) Implementation

This project implements the T5 (Text-to-Text Transfer Transformer) architecture from scratch using PyTorch, following the design described in the "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" paper.

## Features

- Complete T5 architecture implementation
- Relative positional encoding with learned bias
- T5-style layer normalization (RMSNorm)
- Shared self-attention and cross-attention mechanisms
- Text generation capabilities
- Weight tying between embedding and output layers

## Technical Details

- **Architecture**: Configurable encoder-decoder with relative position embeddings
- **Default Configuration**: 6-layer encoder, 6-layer decoder, 8-head attention
- **Implementation**: Pure PyTorch with no transformer-specific libraries
- **Training**: Supports text-to-text format for all NLP tasks

## Usage

The implementation can be used for:
- Text classification
- Question answering
- Text summarization
- Machine translation
- Any NLP task framed as text-to-text

## Requirements

```
torch>=1.7.0
numpy>=1.19.0
```

## Example

```python
# Initialize model
model = T5Model(
    vocab_size=32128,
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048
)

# Create inputs
input_ids = tokenizer.encode("translate English to German: How are you?", return_tensors="pt")
encoder_attention_mask = torch.ones_like(input_ids)

# Generate text
generated_ids = model.generate(
    input_ids,
    encoder_attention_mask=encoder_attention_mask,
    max_length=50,
    bos_token_id=0,
    eos_token_id=1
)

# Decode output
output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
```