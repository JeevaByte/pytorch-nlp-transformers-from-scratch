import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RobertaEmbeddings(nn.Module):
    """RoBERTa embeddings with token, position, and token type embeddings"""
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout_prob=0.1):
        super(RobertaEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=1)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Position IDs buffer (for efficiency)
        self.register_buffer(
            "position_ids", 
            torch.arange(max_position_embeddings).expand((1, -1))
        )
        
    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
            
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class RobertaSelfAttention(nn.Module):
    """Multi-head self-attention for RoBERTa"""
    def __init__(self, hidden_size, num_attention_heads, dropout_prob=0.1):
        super(RobertaSelfAttention, self).__init__()
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, Key, Value projections
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Output projection
        self.output = nn.Linear(hidden_size, hidden_size)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_prob)
        
    def transpose_for_scores(self, x):
        """Reshape from [batch_size, seq_length, hidden_size] to [batch_size, num_heads, seq_length, head_size]"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        # Project query, key, value
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Apply softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to value layer
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back to [batch_size, seq_length, hidden_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        # Apply output projection
        output = self.output(context_layer)
        
        return output

class RobertaSelfOutput(nn.Module):
    """Output of self-attention with residual connection and layer normalization"""
    def __init__(self, hidden_size, dropout_prob=0.1):
        super(RobertaSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states

class RobertaAttention(nn.Module):
    """Complete attention module with self-attention and output layers"""
    def __init__(self, hidden_size, num_attention_heads, dropout_prob=0.1):
        super(RobertaAttention, self).__init__()
        self.self_attention = RobertaSelfAttention(hidden_size, num_attention_heads, dropout_prob)
        self.output = RobertaSelfOutput(hidden_size, dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None):
        self_output = self.self_attention(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output

class RobertaIntermediate(nn.Module):
    """Intermediate feed-forward layer"""
    def __init__(self, hidden_size, intermediate_size):
        super(RobertaIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.gelu = nn.GELU()
        
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states

class RobertaOutput(nn.Module):
    """Output of feed-forward layer with residual connection and layer normalization"""
    def __init__(self, hidden_size, intermediate_size, dropout_prob=0.1):
        super(RobertaOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states

class RobertaLayer(nn.Module):
    """Complete RoBERTa layer with attention and feed-forward"""
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout_prob=0.1):
        super(RobertaLayer, self).__init__()
        self.attention = RobertaAttention(hidden_size, num_attention_heads, dropout_prob)
        self.intermediate = RobertaIntermediate(hidden_size, intermediate_size)
        self.output = RobertaOutput(hidden_size, intermediate_size, dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class RobertaEncoder(nn.Module):
    """Stack of RoBERTa layers"""
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, num_hidden_layers, dropout_prob=0.1):
        super(RobertaEncoder, self).__init__()
        self.layers = nn.ModuleList([
            RobertaLayer(hidden_size, num_attention_heads, intermediate_size, dropout_prob)
            for _ in range(num_hidden_layers)
        ])
        
    def forward(self, hidden_states, attention_mask=None):
        all_encoder_layers = []
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
            
        return all_encoder_layers

class RobertaPooler(nn.Module):
    """Pooler for sentence-level tasks"""
    def __init__(self, hidden_size):
        super(RobertaPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states):
        # Take the first token (CLS) representation
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class RobertaModel(nn.Module):
    """Base RoBERTa model"""
    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12, 
                 num_attention_heads=12, intermediate_size=3072, 
                 max_position_embeddings=514, type_vocab_size=1, dropout_prob=0.1):
        super(RobertaModel, self).__init__()
        
        self.embeddings = RobertaEmbeddings(
            vocab_size, 
            hidden_size, 
            max_position_embeddings, 
            type_vocab_size, 
            dropout_prob
        )
        
        self.encoder = RobertaEncoder(
            hidden_size, 
            num_attention_heads, 
            intermediate_size, 
            num_hidden_layers, 
            dropout_prob
        )
        
        self.pooler = RobertaPooler(hidden_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights similar to RoBERTa"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # Create extended attention mask for transformer
        # [batch_size, 1, 1, seq_length]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Convert mask to additive mask
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Get embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        
        # Apply encoder
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = encoder_outputs[-1]
        
        # Apply pooler
        pooled_output = self.pooler(sequence_output)
        
        return sequence_output, pooled_output

class RobertaForSequenceClassification(nn.Module):
    """RoBERTa for sequence classification tasks"""
    def __init__(self, vocab_size, num_labels, hidden_size=768, num_hidden_layers=12, 
                 num_attention_heads=12, intermediate_size=3072, 
                 max_position_embeddings=514, type_vocab_size=1, dropout_prob=0.1):
        super(RobertaForSequenceClassification, self).__init__()
        
        self.roberta = RobertaModel(
            vocab_size, 
            hidden_size, 
            num_hidden_layers, 
            num_attention_heads, 
            intermediate_size, 
            max_position_embeddings, 
            type_vocab_size, 
            dropout_prob
        )
        
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, labels=None):
        _, pooled_output = self.roberta(input_ids, token_type_ids, attention_mask, position_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits

class RobertaForTokenClassification(nn.Module):
    """RoBERTa for token classification tasks (e.g., NER)"""
    def __init__(self, vocab_size, num_labels, hidden_size=768, num_hidden_layers=12, 
                 num_attention_heads=12, intermediate_size=3072, 
                 max_position_embeddings=514, type_vocab_size=1, dropout_prob=0.1):
        super(RobertaForTokenClassification, self).__init__()
        
        self.num_labels = num_labels
        
        self.roberta = RobertaModel(
            vocab_size, 
            hidden_size, 
            num_hidden_layers, 
            num_attention_heads, 
            intermediate_size, 
            max_position_embeddings, 
            type_vocab_size, 
            dropout_prob
        )
        
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, labels=None):
        sequence_output, _ = self.roberta(input_ids, token_type_ids, attention_mask, position_ids)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                
            return loss, logits
        else:
            return logits

class RobertaForQuestionAnswering(nn.Module):
    """RoBERTa for question answering tasks"""
    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12, 
                 num_attention_heads=12, intermediate_size=3072, 
                 max_position_embeddings=514, type_vocab_size=1, dropout_prob=0.1):
        super(RobertaForQuestionAnswering, self).__init__()
        
        self.roberta = RobertaModel(
            vocab_size, 
            hidden_size, 
            num_hidden_layers, 
            num_attention_heads, 
            intermediate_size, 
            max_position_embeddings, 
            type_vocab_size, 
            dropout_prob
        )
        
        self.qa_outputs = nn.Linear(hidden_size, 2)  # 2 for start/end position
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, 
                start_positions=None, end_positions=None):
        sequence_output, _ = self.roberta(input_ids, token_type_ids, attention_mask, position_ids)
        
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
                
            # Sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss, start_logits, end_logits
        else:
            return start_logits, end_logits

# Example usage
if __name__ == "__main__":
    # Small model for demonstration
    vocab_size = 50265  # Standard RoBERTa vocabulary size
    batch_size = 2
    seq_length = 16
    
    # Initialize model (smaller than standard RoBERTa for demonstration)
    model = RobertaForSequenceClassification(
        vocab_size=vocab_size,
        num_labels=2,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072
    )
    
    # Create dummy inputs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    
    # Forward pass
    logits = model(input_ids, attention_mask=attention_mask)
    print(f"Output shape: {logits.shape}")  # Should be [batch_size, num_labels]