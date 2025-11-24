import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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

def train_decision_transformer(model, train_loader, optimizer, device, num_epochs=50):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Mover datos al dispositivo
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            rtg = batch['rtg'].to(device)
            timesteps = batch['timesteps'].to(device)
            groups = batch['groups'].to(device)
            targets = batch['targets'].to(device)
            
            # Forward pass
            logits = model(states, actions, rtg, timesteps, groups)
            
            # Calcular pérdida solo en posiciones válidas
            loss = F.cross_entropy(
                logits.transpose(1, 2),  # (B, C, L) para cross_entropy
                targets,
                ignore_index=-1
            )
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    return model

from torch.utils.data import Dataset, DataLoader

class RecommendationDataset(Dataset):
    def __init__(self, trajectories, context_length=20):
        """
        Dataset para Decision Transformer en recomendaciones.
        
        Args:
            trajectories: Lista de diccionarios con las trayectorias de usuarios
            context_length: Longitud máxima de la secuencia de contexto
        """
        self.trajectories = trajectories
        self.context_length = context_length
        
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        
        # Extraer datos de la trayectoria
        items = traj['items']  # secuencia de items
        ratings = traj.get('ratings', np.ones(len(items)))  # ratings (opcional)
        rtg = traj['returns_to_go']  # return-to-go acumulado
        timesteps = traj['timesteps']  # pasos de tiempo
        group = traj['user_group']  # grupo de usuario
        
        seq_len = min(len(items), self.context_length)
        
        # Seleccionar ventana aleatoria si la secuencia es larga
        if len(items) > self.context_length:
            start_idx = np.random.randint(0, len(items) - self.context_length)
        else:
            start_idx = 0
        
        end_idx = start_idx + seq_len
        
        # States: items hasta el tiempo t
        states = items[start_idx:end_idx]
        
        # Actions: próximo item (en t+1)
        actions = np.full(seq_len, -1, dtype=np.int64)  # padding por defecto
        if seq_len > 1:
            actions[:-1] = items[start_idx+1:end_idx]
        
        # Targets: igual que actions (próximo item a predecir)
        targets = actions.copy()
        
        # Returns-to-go (normalizado)
        rtg_seq = rtg[start_idx:end_idx].reshape(-1, 1)
        
        # Timesteps (relativos al inicio de la ventana)
        time_seq = timesteps[start_idx:end_idx] - start_idx
        
        return {
            'states': torch.tensor(states, dtype=torch.long),
            'actions': torch.tensor(actions, dtype=torch.long),
            'rtg': torch.tensor(rtg_seq, dtype=torch.float32),
            'timesteps': torch.tensor(time_seq, dtype=torch.long),
            'groups': torch.tensor(group, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long)
        }