import torch
import torch.nn as nn
import torch.nn.functional as F

class DecisionTransformer(nn.Module):
    def __init__(
            self,
            num_items = 752,
            num_groups = 8,
            hidden_dim = 128,
            n_layers = 3,
            n_heads = 8,
            contxt_len = 20,
            max_timesteps = 200,
            dropout = 0.1, 
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.contxt_len = contxt_len

        # creamos las capas de embedding

        # item_embedding: Embedding para los IDs de los items (películas).
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        # group_embedding: Embedding para los grupos de usuarios.
        self.group_embedding = nn.Embedding(num_groups, hidden_dim)
        # rtg_embedding: Embedding para los retornos a futuro (returns-to-go).
        self.rtg_embedding = nn.Linear(1, hidden_dim)
        # timestep_embedding: Embedding para los pasos temporales.
        self.timestep_embedding = nn.Embedding(max_timesteps, hidden_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4*hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # Capa final para predecir la calificación (rating) del siguiente item.
        self.predict_item = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_items)
        )

        # Normalización
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                states,
                actions,
                returns_to_go,
                timesteps,
                groups,
                attention_mask=None 
    ):
        '''
        Parámetros:
        states: Secuencia de estados, i.e., IDs de items en history).
        actions: IDs de items recomendados (targets) .
        returns_to_go: Secuencia de retornos futuros esperados.
        timesteps: Secuencia de posiciones temporales.
        groups: cluster del usuario.
        attention_mask: Máscara para la atención del Transformer (opcional).

        Returns:
        logits: Predicciones de ratings para los próximos items.
        '''

        batch_size, seq_length = states.shape

        # Embeddings --> Dimensiones [batch_size, sequence_length, embedding_dimension].

        # Embedding de items (estados)
        state_embeddings = self.item_embedding(states)  # (B, L, D)

        # Embedding de acciones (items recomendados)
        action_embeddings = self.item_embedding(actions)  # (B, L, D)

        # Embedding de returns-to-go
        rtg_embeddings = self.rtg_embedding(returns_to_go)  # (B, L, D)

        # Embedding de timesteps
        time_embeddings = self.timestep_embedding(timesteps)  # (B, L, D)

        # Embedding de grupo (expandido a toda la secuencia)
        group_embeddings = self.group_embedding(groups).unsqueeze(1).expand(-1, seq_length, -1)  # (B, L, D)

        # Sumar embeddings

        h = state_embeddings + rtg_embeddings + time_embeddings + group_embeddings
        h = self.ln(h)

        # === CAUSAL MASK ===
        # Asegurar que cada timestep solo ve el pasado
        if attention_mask is None:
            attention_mask = self._generate_causal_mask(seq_length).to(h.device)
        
        # === TRANSFORMER ===
        h = self.transformer(h, mask=attention_mask)  # (B, L, H)
        
        # === PREDICT NEXT ITEM ===
        item_logits = self.predict_item(h)  # (B, L, num_items)
        
        return item_logits
    
    def _generate_causal_mask(self, seq_len):
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask
    
def train_decision_transformer(
    model, 
    train_loader, 
    optimizer, 
    device,
    num_epochs=50
):
    """
    Entrena el Decision Transformer.
    
    Loss: Cross-entropy entre item predicho y item verdadero
    """
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in train_loader:
            states = batch['states'].to(device)      # (B, L)
            actions = batch['actions'].to(device)    # (B, L)
            rtg = batch['rtg'].to(device)            # (B, L, 1)
            timesteps = batch['timesteps'].to(device) # (B, L)
            groups = batch['groups'].to(device)      # (B,)
            targets = batch['targets'].to(device)    # (B, L) - next items
            
            # Forward pass
            logits = model(states, actions, rtg, timesteps, groups)
            
            # Compute loss
            loss = F.cross_entropy(
                logits.reshape(-1, model.num_items),
                targets.reshape(-1),
                ignore_index=-1  # para padding
            )
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    
    return model

from torch.utils.data import Dataset, DataLoader

class RecommendationDataset(Dataset):
    def __init__(self, trajectories, context_length=20):
        self.trajectories = trajectories
        self.context_length = context_length
        
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        
        # Extraer secuencia completa
        items = traj['items']
        ratings = traj['ratings']
        rtg = traj['returns_to_go']
        timesteps = traj['timesteps']
        group = traj['user_group']
        
        # Tomar una ventana de context_length
        # (o toda la secuencia si es más corta)
        seq_len = min(len(items), self.context_length)
        
        # Random start point (para data augmentation)
        if len(items) > self.context_length:
            start_idx = np.random.randint(0, len(items) - self.context_length + 1)
        else:
            start_idx = 0
        
        end_idx = start_idx + seq_len
        
        # States: items vistos (history)
        # Para t, state = items[:t]
        states = items[start_idx:end_idx]
        
        # Actions: items que fueron "recomendados" (mismo que states shifted)
        actions = items[start_idx:end_idx]
        
        # Targets: próximo item a predecir
        targets = np.zeros(seq_len, dtype=np.int64)
        targets[:-1] = items[start_idx+1:end_idx]
        targets[-1] = -1  # padding para último timestep
        
        # Returns-to-go
        rtg_seq = rtg[start_idx:end_idx].reshape(-1, 1)
        
        # Timesteps
        time_seq = timesteps[start_idx:end_idx]
        
        return {
            'states': torch.tensor(states, dtype=torch.long),
            'actions': torch.tensor(actions, dtype=torch.long),
            'rtg': torch.tensor(rtg_seq, dtype=torch.float32),
            'timesteps': torch.tensor(time_seq, dtype=torch.long),
            'groups': torch.tensor(group, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long)
        }
    

