import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DecisionTransformer(nn.Module):
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
        
        # === EMBEDDINGS SEPARADOS ===
        
        # Embeddings independientes para cada tipo de token
        self.state_embedding = nn.Embedding(num_items, hidden_dim)
        self.action_embedding = nn.Embedding(num_items, hidden_dim)
        self.rtg_embedding = nn.Linear(1, hidden_dim)
        
        # Embeddings de contexto
        self.group_embedding = nn.Embedding(num_groups, hidden_dim)
        self.timestep_embedding = nn.Embedding(max_timestep, hidden_dim)
        
        # Token type embeddings (RTG, STATE, ACTION)
        self.token_type_embeddings = nn.Embedding(3, hidden_dim)
        
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
        
        # === PREDICTION HEADS ===
        self.predict_item = nn.Linear(hidden_dim, num_items)
        self.ln = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, 
                states,        # (B, L)
                actions,       # (B, L) 
                returns_to_go, # (B, L, 1)
                timesteps,     # (B, L)
                user_groups,   # (B,)
                attention_mask=None):
        
        batch_size, seq_len = states.shape
        device = states.device

        # === CREAR SECUENCIA INTERLEAVED ===
        # RTG, STATE, ACTION para cada timestep
        rtg_emb = self.rtg_embedding(returns_to_go)  # (B, L, H)
        
        # State embedding (con padding handling)
        state_emb = self.state_embedding(states)  # (B, L, H)
        
        # Action embedding (manejar padding en actions)
        actions_safe = actions.clone()
        actions_safe[actions == -1] = 0  # usar 0 para padding
        action_emb = self.action_embedding(actions_safe)  # (B, L, H)
        
        # === INTERLEAVING: [RTG_0, STATE_0, ACTION_0, RTG_1, STATE_1, ACTION_1, ...] ===
        tokens_list = []
        for t in range(seq_len):
            tokens_list.extend([rtg_emb[:, t:t+1, :],    # (B, 1, H)
                              state_emb[:, t:t+1, :],   # (B, 1, H)  
                              action_emb[:, t:t+1, :]]) # (B, 1, H)
        
        # Concatenar en la dimensión de secuencia
        h = torch.cat(tokens_list, dim=1)  # (B, L*3, H)
        
        # === AÑADIR EMBEDDINGS DE POSICIÓN Y TIPO ===
        
        # Token type embeddings: 0=RTG, 1=STATE, 2=ACTION
        token_types = []
        for t in range(seq_len):
            token_types.extend([0, 1, 2])
        token_types = torch.tensor(token_types, device=device).unsqueeze(0).expand(batch_size, -1)  # (B, L*3)
        token_type_emb = self.token_type_embeddings(token_types)  # (B, L*3, H)
        h = h + token_type_emb
        
        # Timestep embeddings (expandir para los 3 tokens por timestep)
        timestep_emb = self.timestep_embedding(timesteps)  # (B, L, H)
        timestep_emb_interleaved = timestep_emb.repeat_interleave(3, dim=1)  # (B, L*3, H)
        h = h + timestep_emb_interleaved
        
        # Group embeddings
        group_emb = self.group_embedding(user_groups)  # (B, H)
        group_emb = group_emb.unsqueeze(1).expand(-1, seq_len * 3, -1)  # (B, L*3, H)
        h = h + group_emb
        
        # Layer norm
        h = self.ln(h)
        
        # === MÁSCARA DE ATENCIÓN CAUSAL ===
        if attention_mask is None:
            seq_len_interleaved = seq_len * 3
            attention_mask = self._generate_causal_mask(seq_len_interleaved).to(device)
        
        # === TRANSFORMER ===
        h = self.transformer(h, mask=attention_mask)  # (B, L*3, H)
        
        # === PREDICCIÓN: Solo en posiciones de STATE ===
        # Las posiciones de STATE son: 1, 4, 7, ... (índices 3*t + 1)
        state_positions = torch.arange(seq_len, device=device) * 3 + 1
        h_states = h[:, state_positions, :]  # (B, L, H)
        
        item_logits = self.predict_item(h_states)  # (B, L, num_items)
        
        return item_logits
    
    def _generate_causal_mask(self, seq_len):
        """Máscara causal que respeta la estructura RTG->STATE->ACTION"""
        mask = torch.ones(seq_len, seq_len) * float('-inf')
        
        for i in range(seq_len):
            current_timestep = i // 3
            current_type = i % 3  # 0=RTG, 1=STATE, 2=ACTION
            
            for j in range(i + 1):
                prev_timestep = j // 3
                prev_type = j % 3
                
                # Permitir:
                # - Todos los tokens de timesteps anteriores
                # - En el mismo timestep: RTG puede ver RTG, STATE puede ver RTG y STATE, 
                #   ACTION puede ver todos del mismo timestep
                if prev_timestep < current_timestep:
                    mask[i, j] = 0
                elif prev_timestep == current_timestep:
                    if prev_type <= current_type:
                        mask[i, j] = 0
        
        return mask
