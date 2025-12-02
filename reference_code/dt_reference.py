import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DecisionTransformerReference(nn.Module):
    def __init__(
        self,
        num_items=752,
        num_groups=8,
        hidden_dim=128,
        n_layers=3,
        n_heads=4,
        context_length=20,
        max_timestep=200,
        dropout=0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.context_length = context_length
        self.num_items = num_items
        # === EMBEDDINGS ===
        
        # Item embedding (para history y acciones)
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        
        # User group embedding
        self.group_embedding = nn.Embedding(num_groups, hidden_dim)
        
        # Return-to-go embedding (escalar continuo)
        self.rtg_embedding = nn.Linear(1, hidden_dim)
        
        # Timestep embedding (positional encoding)
        self.timestep_embedding = nn.Embedding(max_timestep, hidden_dim)
        
        # === TRANSFORMER ===
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # === PREDICTION HEAD ===
        
        # Predecir qué item recomendar
        self.predict_item = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_items)
        )
        
        # Layer normalization
        self.ln = nn.LayerNorm(hidden_dim)
        
    def forward(self, 
                states,        # (batch, seq_len) - item IDs vistos
                actions,       # (batch, seq_len) - item IDs recomendados
                returns_to_go, # (batch, seq_len, 1) - R̂ values
                timesteps,     # (batch, seq_len) - posiciones temporales
                user_groups,   # (batch,) - cluster del usuario
                attention_mask=None):
        """
        Args:
            states: IDs de items en history
            actions: IDs de items recomendados (targets)
            returns_to_go: R̂ para cada timestep
            timesteps: posiciones temporales
            user_groups: cluster del usuario
            
        Returns:
            item_logits: (batch, seq_len, num_items) - probabilidades sobre items
        """
        batch_size, seq_len = states.shape
        
        # === EMBED INPUTS ===
        
        # States (history)
        state_emb = self.item_embedding(states)  # (B, L, H)
        
        # Actions (ya recomendados, para autoregression)
        action_emb = self.item_embedding(actions)  # (B, L, H)
        
        # Returns-to-go
        rtg_emb = self.rtg_embedding(returns_to_go)  # (B, L, H)
        
        # Timesteps
        time_emb = self.timestep_embedding(timesteps)  # (B, L, H)
        
        # User group (broadcast a toda la secuencia)
        group_emb = self.group_embedding(user_groups).unsqueeze(1)  # (B, 1, H)
        group_emb = group_emb.expand(-1, seq_len, -1)  # (B, L, H)
        
        # === INTERLEAVE EMBEDDINGS ===
        # Formato: [rtg_0, state_0, action_0, rtg_1, state_1, action_1, ...]
        
        # Para simplicidad, usamos sum de embeddings + positional
        # (En la versión completa, se pueden interleave explícitamente)
        h = state_emb + rtg_emb + time_emb + group_emb
        h = self.ln(h)
        
        # === CAUSAL MASK ===
        # Asegurar que cada timestep solo ve el pasado
        if attention_mask is None:
            attention_mask = self._generate_causal_mask(seq_len).to(h.device)
        
        # === TRANSFORMER ===
        h = self.transformer(h, mask=attention_mask)  # (B, L, H)
        
        # === PREDICT NEXT ITEM ===
        item_logits = self.predict_item(h)  # (B, L, num_items)
        
        return item_logits
    
    def _generate_causal_mask(self, seq_len):
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask