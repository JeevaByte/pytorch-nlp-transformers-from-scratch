import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

def _get_clones(module, N):
    """Produce N identical layers"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionalEncoding(nn.Module):
    """Relative positional encoding for T5"""
    def __init__(self, d_model, num_heads, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_len = max_len
        
        # Create relative position bias table
        self.relative_attention_bias = nn.Embedding(32, num_heads)
        
    def _relative_position_bucket(self, relative_position):
        """Compute relative position buckets for bias"""
        num_buckets = 32
        max_distance = self.max_len
        
        relative_buckets = 0
        
        # Use smaller buckets for small relative positions
        relative_buckets += (relative_position > 0).long() * num_buckets // 2
        relative_buckets += (relative_position < 0).long() * num_buckets // 2
        
        relative_position = torch.abs(relative_position)
        
        # Determine bucket based on log of relative position
        max_exact = num_buckets // 4
        is_small = relative_position < max_exact
        
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets // 2 - max_exact)
        ).long()
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets // 2 - 1)
        )
        
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
    
    def compute_bias(self, seq_length):
        """Compute positional bias for attention"""
        context_position = torch.arange(seq_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(seq_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        
        # Get relative position buckets
        relative_position_bucket = self._relative_position_bucket(relative_position)
        
        # Get embeddings for each bucket
        values = self.relative_attention_bias(relative_position_bucket)
        
        # Reshape to [seq_length, seq_length, num_heads]
        values = values.permute(2, 0, 1)
        return values

class T5Attention(nn.Module):
    """Multi-head attention for T5"""
    def __init__(self, d_model, num_heads, dropout=0.1, has_relative_attention_bias=False):
        super(T5Attention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.has_relative_attention_bias = has_relative_attention_bias
        if has_relative_attention_bias:
            self.relative_attention_bias = PositionalEncoding(d_model, num_heads)
        
    def forward(self, query, key, value, mask=None, position_bias=None):
        batch_size = query.size(0)
        q_seq_len = query.size(1)
        k_seq_len = key.size(1)
        
        # Linear projections and reshape
        q = self.q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add relative position bias if needed
        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.relative_attention_bias.compute_bias(max(q_seq_len, k_seq_len))
            # Slice to the correct size if needed
            if position_bias.size(-1) > k_seq_len or position_bias.size(-2) > q_seq_len:
                position_bias = position_bias[:, :q_seq_len, :k_seq_len]
            position_bias = position_bias.unsqueeze(0)  # Add batch dimension
            
        if position_bias is not None:
            # Ensure position_bias has the right shape for broadcasting
            if position_bias.size(0) == 1:
                position_bias = position_bias.expand(batch_size, -1, -1, -1)
            scores = scores + position_bias
            
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
            
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.o(context)
        
        return output, attn_weights, position_bias

class T5FeedForward(nn.Module):
    """Feed-forward network for T5"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(T5FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))

class T5LayerNorm(nn.Module):
    """T5-style layer normalization (RMSNorm)"""
    def __init__(self, d_model, eps=1e-6):
        super(T5LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        
    def forward(self, x):
        # T5 uses RMSNorm (root mean square normalization)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class T5EncoderLayer(nn.Module):
    """Encoder layer for T5"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, has_relative_attention_bias=False):
        super(T5EncoderLayer, self).__init__()
        
        self.self_attn = T5Attention(d_model, num_heads, dropout, has_relative_attention_bias)
        self.feed_forward = T5FeedForward(d_model, d_ff, dropout)
        
        self.layer_norm1 = T5LayerNorm(d_model)
        self.layer_norm2 = T5LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, position_bias=None):
        # Layer normalization and self-attention
        norm_x = self.layer_norm1(x)
        attn_output, _, position_bias = self.self_attn(norm_x, norm_x, norm_x, mask, position_bias)
        
        # Residual connection and dropout
        x = x + self.dropout(attn_output)
        
        # Layer normalization and feed-forward
        norm_x = self.layer_norm2(x)
        ff_output = self.feed_forward(norm_x)
        
        # Residual connection and dropout
        x = x + self.dropout(ff_output)
        
        return x, position_bias

class T5DecoderLayer(nn.Module):
    """Decoder layer for T5"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, has_relative_attention_bias=False):
        super(T5DecoderLayer, self).__init__()
        
        self.self_attn = T5Attention(d_model, num_heads, dropout, has_relative_attention_bias)
        self.cross_attn = T5Attention(d_model, num_heads, dropout, False)
        self.feed_forward = T5FeedForward(d_model, d_ff, dropout)
        
        self.layer_norm1 = T5LayerNorm(d_model)
        self.layer_norm2 = T5LayerNorm(d_model)
        self.layer_norm3 = T5LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, self_mask=None, cross_mask=None, self_position_bias=None, cross_position_bias=None):
        # Layer normalization and self-attention
        norm_x = self.layer_norm1(x)
        self_attn_output, _, self_position_bias = self.self_attn(norm_x, norm_x, norm_x, self_mask, self_position_bias)
        
        # Residual connection and dropout
        x = x + self.dropout(self_attn_output)
        
        # Layer normalization and cross-attention
        norm_x = self.layer_norm2(x)
        cross_attn_output, _, cross_position_bias = self.cross_attn(norm_x, enc_output, enc_output, cross_mask, cross_position_bias)
        
        # Residual connection and dropout
        x = x + self.dropout(cross_attn_output)
        
        # Layer normalization and feed-forward
        norm_x = self.layer_norm3(x)
        ff_output = self.feed_forward(norm_x)
        
        # Residual connection and dropout
        x = x + self.dropout(ff_output)
        
        return x, self_position_bias, cross_position_bias

class T5Encoder(nn.Module):
    """T5 Encoder"""
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        super(T5Encoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # First layer has relative position bias
        self.layers = nn.ModuleList([
            T5EncoderLayer(d_model, num_heads, d_ff, dropout, has_relative_attention_bias=(i == 0))
            for i in range(num_layers)
        ])
        
        self.final_layer_norm = T5LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask=None):
        # Embedding
        x = self.embedding(input_ids)
        x = self.dropout(x)
        
        # Create attention mask if provided
        if attention_mask is not None:
            # Convert mask of 1s and 0s to mask of 0s and -1e9
            attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        
        # Initialize position bias
        position_bias = None
        
        # Apply encoder layers
        for layer in self.layers:
            x, position_bias = layer(x, attention_mask, position_bias)
            
        # Final layer norm
        x = self.final_layer_norm(x)
        
        return x

class T5Decoder(nn.Module):
    """T5 Decoder"""
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        super(T5Decoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # First layer has relative position bias
        self.layers = nn.ModuleList([
            T5DecoderLayer(d_model, num_heads, d_ff, dropout, has_relative_attention_bias=(i == 0))
            for i in range(num_layers)
        ])
        
        self.final_layer_norm = T5LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        # Embedding
        x = self.embedding(input_ids)
        x = self.dropout(x)
        
        # Create attention masks if provided
        if self_attention_mask is not None:
            # Create causal mask for decoder self-attention
            seq_length = input_ids.size(1)
            causal_mask = torch.tril(torch.ones(seq_length, seq_length, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
            self_attention_mask = self_attention_mask.unsqueeze(1).unsqueeze(2) * causal_mask
            # Convert mask of 1s and 0s to mask of 0s and -1e9
            self_attention_mask = (1.0 - self_attention_mask) * -1e9
            
        if cross_attention_mask is not None:
            # Convert mask of 1s and 0s to mask of 0s and -1e9
            cross_attention_mask = (1.0 - cross_attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        
        # Initialize position biases
        self_position_bias = None
        cross_position_bias = None
        
        # Apply decoder layers
        for layer in self.layers:
            x, self_position_bias, cross_position_bias = layer(
                x, encoder_output, self_attention_mask, cross_attention_mask, 
                self_position_bias, cross_position_bias
            )
            
        # Final layer norm
        x = self.final_layer_norm(x)
        
        return x

class T5Model(nn.Module):
    """T5 Model"""
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1):
        super(T5Model, self).__init__()
        
        self.encoder = T5Encoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout)
        self.decoder = T5Decoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout)
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights between embedding and output layer
        self.lm_head.weight = self.decoder.embedding.weight
        
    def forward(self, input_ids, decoder_input_ids, encoder_attention_mask=None, decoder_attention_mask=None):
        # Encode input sequence
        encoder_output = self.encoder(input_ids, encoder_attention_mask)
        
        # Decode output sequence
        decoder_output = self.decoder(
            decoder_input_ids, 
            encoder_output, 
            decoder_attention_mask, 
            encoder_attention_mask
        )
        
        # Project to vocabulary
        lm_logits = self.lm_head(decoder_output)
        
        return lm_logits
    
    def generate(self, input_ids, encoder_attention_mask=None, max_length=20, 
                 bos_token_id=0, eos_token_id=1, pad_token_id=0):
        """Generate text using the model"""
        with torch.no_grad():
            batch_size = input_ids.size(0)
            
            # Encode input sequence
            encoder_output = self.encoder(input_ids, encoder_attention_mask)
            
            # Initialize decoder input with BOS token
            decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device) * bos_token_id
            
            # Generate tokens one by one
            for _ in range(max_length - 1):
                # Create decoder attention mask
                decoder_attention_mask = torch.ones_like(decoder_input_ids)
                
                # Decode current sequence
                decoder_output = self.decoder(
                    decoder_input_ids, 
                    encoder_output, 
                    decoder_attention_mask, 
                    encoder_attention_mask
                )
                
                # Get next token prediction
                next_token_logits = self.lm_head(decoder_output[:, -1, :])
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append next token to sequence
                decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
                
                # Stop if all sequences have generated EOS token
                if (next_token == eos_token_id).all():
                    break
            
            return decoder_input_ids

# Example usage
if __name__ == "__main__":
    # Small model for demonstration
    vocab_size = 32128  # Standard T5 vocabulary size
    batch_size = 2
    seq_length = 16
    
    # Initialize model (smaller than standard T5 for demonstration)
    model = T5Model(
        vocab_size=vocab_size,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048
    )
    
    # Create dummy inputs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    decoder_input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Create attention masks
    encoder_attention_mask = torch.ones_like(input_ids)
    decoder_attention_mask = torch.ones_like(decoder_input_ids)
    
    # Forward pass
    logits = model(input_ids, decoder_input_ids, encoder_attention_mask, decoder_attention_mask)
    print(f"Output shape: {logits.shape}")  # Should be [batch_size, seq_length, vocab_size]
    
    # Generate text
    generated = model.generate(input_ids, encoder_attention_mask)
    print(f"Generated sequence shape: {generated.shape}")  # Should be [batch_size, seq_length]