import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer model"""
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter but should be saved and loaded with the model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """Multi-head attention for transformer"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape
        q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(context)
        
        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    """Encoder layer for transformer"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer normalization
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    """Decoder layer for transformer"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self-attention with residual connection and layer normalization
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with residual connection and layer normalization
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class Encoder(nn.Module):
    """Transformer encoder"""
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout=0.1):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, src, src_mask=None):
        # Embed tokens and add positional encoding
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        
        # Apply encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)
            
        return self.norm(x)

class Decoder(nn.Module):
    """Transformer decoder"""
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout=0.1):
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        # Embed tokens and add positional encoding
        x = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        
        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
            
        return self.norm(x)

class Transformer(nn.Module):
    """Transformer model for neural machine translation"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, 
                 num_heads=8, d_ff=2048, max_seq_length=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)
        
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode source sequence
        enc_output = self.encoder(src, src_mask)
        
        # Decode target sequence
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        output = self.output_layer(dec_output)
        
        return output
    
    def create_masks(self, src, tgt):
        # Source mask (padding mask)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # Target mask (padding and subsequent mask)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        
        # Create subsequent mask to prevent attending to future positions
        seq_length = tgt.size(1)
        subsequent_mask = torch.triu(torch.ones((seq_length, seq_length), device=tgt.device), diagonal=1) == 0
        subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(0)
        
        # Combine padding mask and subsequent mask
        tgt_mask = tgt_mask & subsequent_mask
        
        return src_mask, tgt_mask
    
    def translate(self, src, max_length=100, sos_idx=2, eos_idx=3):
        """Translate source sequence to target sequence"""
        with torch.no_grad():
            # Create source mask
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
            
            # Encode source sequence
            enc_output = self.encoder(src, src_mask)
            
            # Initialize target sequence with SOS token
            tgt = torch.ones(src.size(0), 1).fill_(sos_idx).long().to(src.device)
            
            for i in range(max_length - 1):
                # Create target mask
                tgt_mask = self.create_masks(src, tgt)[1]
                
                # Decode target sequence
                dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
                
                # Project to vocabulary and get next token
                prob = self.output_layer(dec_output[:, -1])
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.unsqueeze(1)
                
                # Append next token to target sequence
                tgt = torch.cat([tgt, next_word], dim=1)
                
                # Stop if EOS token is generated
                if (next_word == eos_idx).sum() == src.size(0):
                    break
                    
            return tgt

# Example usage
if __name__ == "__main__":
    # Small model for demonstration
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    batch_size = 8
    src_seq_length = 20
    tgt_seq_length = 15
    
    # Initialize model
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048
    )
    
    # Create dummy inputs
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_length))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_length))
    
    # Create masks
    src_mask, tgt_mask = model.create_masks(src, tgt)
    
    # Forward pass
    output = model(src, tgt, src_mask, tgt_mask)
    print(f"Output shape: {output.shape}")  # Should be [batch_size, tgt_seq_length, tgt_vocab_size]
    
    # Translation
    translated = model.translate(src)
    print(f"Translated sequence shape: {translated.shape}")  # Should be [batch_size, seq_length]