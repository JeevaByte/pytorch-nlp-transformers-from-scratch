import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Make sure mask is broadcastable to attention scores
            attn_scores = attn_scores + mask
            
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        
        return output
        
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        q = self.W_q(Q)
        k = self.W_k(K)
        v = self.W_v(V)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        output = self.W_o(attn_output)
        
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.mha(x, x, x, mask)
        x = self.layernorm1(x + self.dropout(attn_output))
        
        ff_output = self.ff(x)
        x = self.layernorm2(x + self.dropout(ff_output))
        
        return x

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_length, dropout=0.1):
        super(BERTEmbedding, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_length, d_model)
        self.segment_embeddings = nn.Embedding(2, d_model)  # For sentence A/B
        
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, segment_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
            
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        segment_embeddings = self.segment_embeddings(segment_ids)
        
        embeddings = token_embeddings + position_embeddings + segment_embeddings
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BERT(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=12, num_heads=12, 
                 d_ff=3072, max_seq_length=512, dropout=0.1):
        super(BERT, self).__init__()
        
        self.embeddings = BERTEmbedding(vocab_size, d_model, max_seq_length, dropout)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.pooler = nn.Linear(d_model, d_model)
        self.pooler_activation = nn.Tanh()
        
    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        # Embedding
        embeddings = self.embeddings(input_ids, segment_ids)
        
        # Prepare attention mask for multi-head attention
        if attention_mask is not None:
            # Expand attention mask to [batch_size, 1, 1, seq_length]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Convert mask of 1s and 0s to mask of 0s and -1e9
            extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
        else:
            extended_attention_mask = None
        
        # Encoder layers
        x = embeddings
        for layer in self.encoder_layers:
            x = layer(x, extended_attention_mask)
            
        # Pooled output for classification tasks
        pooled_output = self.pooler_activation(self.pooler(x[:, 0]))
        
        return x, pooled_output

class BERTForSequenceClassification(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=768, num_layers=12, 
                 num_heads=12, d_ff=3072, max_seq_length=512, dropout=0.1):
        super(BERTForSequenceClassification, self).__init__()
        
        self.bert = BERT(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        _, pooled_output = self.bert(input_ids, segment_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

# Example usage
if __name__ == "__main__":
    # Small model for demonstration
    vocab_size = 30000
    num_classes = 2
    batch_size = 8
    seq_length = 128
    
    # Initialize model
    model = BERTForSequenceClassification(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=768,
        num_layers=6,
        num_heads=12,
        d_ff=3072,
        max_seq_length=512
    )
    
    # Create dummy inputs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    segment_ids = torch.zeros_like(input_ids)
    attention_mask = torch.ones_like(input_ids)
    
    # Forward pass
    logits = model(input_ids, segment_ids, attention_mask)
    print(f"Output shape: {logits.shape}")  # Should be [batch_size, num_classes]