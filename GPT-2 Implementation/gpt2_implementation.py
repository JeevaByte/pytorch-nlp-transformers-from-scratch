import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class LayerNorm(nn.Module):
    """Layer Normalization"""
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias

class Conv1D(nn.Module):
    """1D Convolution for GPT2"""
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

class Attention(nn.Module):
    """Multi-head attention for GPT2"""
    def __init__(self, nx, n_ctx, n_head, scale=False, dropout=0.1):
        super(Attention, self).__init__()
        self.n_head = n_head
        self.n_embd = nx
        self.scale = scale
        
        self.c_attn = Conv1D(3 * nx, nx)
        self.c_proj = Conv1D(nx, nx)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask to ensure that attention is only applied to the left
        mask = torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx)
        self.register_buffer('mask', mask)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
            
        # Apply the causal mask
        w = w.masked_fill(self.mask[:, :, :w.size(-2), :w.size(-1)] == 0, -1e10)
        
        w = F.softmax(w, dim=-1)
        w = self.attn_dropout(w)
        
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x):
        x_shape = x.size()
        
        # Project query, key, value
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Split heads
        q = self.split_heads(q)
        k = self.split_heads(k, k=True)
        v = self.split_heads(v)
        
        # Compute attention
        a = self._attn(q, k, v)
        
        # Merge heads
        a = self.merge_heads(a)
        
        # Project back to residual
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        
        return a

class MLP(nn.Module):
    """MLP for GPT2"""
    def __init__(self, n_state, n_embd, dropout=0.1):
        super(MLP, self).__init__()
        self.c_fc = Conv1D(n_state, n_embd)
        self.c_proj = Conv1D(n_embd, n_state)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)

class Block(nn.Module):
    """GPT2 Block"""
    def __init__(self, n_ctx, n_embd, n_head, dropout=0.1):
        super(Block, self).__init__()
        self.ln_1 = LayerNorm(n_embd)
        self.attn = Attention(n_embd, n_ctx, n_head, scale=True, dropout=dropout)
        self.ln_2 = LayerNorm(n_embd)
        self.mlp = MLP(4 * n_embd, n_embd, dropout=dropout)

    def forward(self, x):
        a = self.attn(self.ln_1(x))
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x

class GPT2Model(nn.Module):
    """GPT-2 Language Model"""
    def __init__(self, vocab_size, n_ctx=1024, n_embd=768, n_layer=12, n_head=12, dropout=0.1):
        super(GPT2Model, self).__init__()
        self.n_embd = n_embd
        self.n_vocab = vocab_size
        self.n_ctx = n_ctx
        
        # Token + Position embeddings
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_ctx, n_embd)
        self.drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.h = nn.ModuleList([Block(n_ctx, n_embd, n_head, dropout) for _ in range(n_layer)])
        
        # Final layer norm
        self.ln_f = LayerNorm(n_embd)

    def forward(self, input_ids, position_ids=None):
        batch_size, seq_length = input_ids.size()
        
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Get embeddings
        token_embeddings = self.wte(input_ids)
        position_embeddings = self.wpe(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeddings + position_embeddings
        hidden_states = self.drop(hidden_states)
        
        # Apply transformer blocks
        for block in self.h:
            hidden_states = block(hidden_states)
        
        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        return hidden_states

class GPT2LMHeadModel(nn.Module):
    """GPT-2 with language modeling head"""
    def __init__(self, vocab_size, n_ctx=1024, n_embd=768, n_layer=12, n_head=12, dropout=0.1):
        super(GPT2LMHeadModel, self).__init__()
        self.transformer = GPT2Model(vocab_size, n_ctx, n_embd, n_layer, n_head, dropout)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie weights between embedding and output layer
        self.lm_head.weight = self.transformer.wte.weight
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, position_ids=None):
        hidden_states = self.transformer(input_ids, position_ids)
        lm_logits = self.lm_head(hidden_states)
        return lm_logits
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=0, top_p=0.9):
        """Generate text using the model with various decoding strategies"""
        with torch.no_grad():
            for _ in range(max_length):
                # Get predictions
                outputs = self(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[0, indices_to_remove] = -float('Inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to input_ids
                input_ids = torch.cat((input_ids, next_token), dim=1)
                
        return input_ids

# Example usage
if __name__ == "__main__":
    # Small model for demonstration
    vocab_size = 50257  # Standard GPT-2 vocabulary size
    batch_size = 2
    seq_length = 16
    
    # Initialize model (smaller than standard GPT-2 for demonstration)
    model = GPT2LMHeadModel(
        vocab_size=vocab_size,
        n_ctx=1024,
        n_embd=768,
        n_layer=6,
        n_head=12
    )
    
    # Create dummy inputs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Forward pass
    logits = model(input_ids)
    print(f"Output shape: {logits.shape}")  # Should be [batch_size, seq_length, vocab_size]
    
    # Generate text
    generated = model.generate(input_ids[:1], max_length=10)
    print(f"Generated sequence shape: {generated.shape}")  # Should be [1, seq_length+10]