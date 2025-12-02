import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

        # === EMBEDDINGS SEPARADOS === #
        self.state_embedding = nn.Embedding(num_items, hidden_dim)
        self.action_embedding = nn.Embedding(num_items, hidden_dim)
        self.rtg_embedding = nn.Linear(1, hidden_dim)

        self.group_embedding = nn.Embedding(num_groups, hidden_dim)
        self.timestep_embedding = nn.Embedding(max_timestep, hidden_dim)
        self.token_type_embeddings = nn.Embedding(3, hidden_dim)  # 0=RTG,1=STATE,2=ACTION

        # === TRANSFORMER === #
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # === PREDICTION HEAD === #
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

    def forward(
        self,
        states,          # (B, L)
        actions,         # (B, L)
        returns_to_go,   # (B, L, 1)
        timesteps,       # (B, L)
        user_groups,     # (B,)
        attention_mask=None
    ):
        B, L = states.shape
        device = states.device

        # === EMBEDDINGS === #
        rtg_emb = self.rtg_embedding(returns_to_go)
        state_emb = self.state_embedding(states)

        # Replace padding -1 with 0 for embedding
        actions_safe = actions.clone()
        actions_safe[actions == -1] = 0
        action_emb = self.action_embedding(actions_safe)

        # === INTERLEAVE: RTG, STATE, ACTION === #
        tokens = []
        for t in range(L):
            tokens.append(rtg_emb[:, t:t+1, :])
            tokens.append(state_emb[:, t:t+1, :])
            tokens.append(action_emb[:, t:t+1, :])

        h = torch.cat(tokens, dim=1)  # (B, 3L, H)

        # === TOKEN TYPE EMBEDDINGS === #
        token_type_ids = []
        for t in range(L):
            token_type_ids.extend([0, 1, 2])  # RTG, STATE, ACTION

        token_type_ids = torch.tensor(token_type_ids, device=device).unsqueeze(0)
        token_type_ids = token_type_ids.expand(B, -1)  # (B, 3L)
        h = h + self.token_type_embeddings(token_type_ids)

        # === TIMESTEP EMBEDDINGS === #
        t_emb = self.timestep_embedding(timesteps)         # (B, L, H)
        t_emb = t_emb.repeat_interleave(3, dim=1)          # (B, 3L, H)
        h = h + t_emb

        # === USER GROUP EMBEDDINGS === #
        g_emb = self.group_embedding(user_groups)          # (B, H)
        g_emb = g_emb.unsqueeze(1).expand(-1, 3*L, -1)      # (B, 3L, H)
        h = h + g_emb

        # === NORMALIZATION === #
        h = self.ln(h)

        # === CAUSAL MASK === #
        if attention_mask is None:
            attention_mask = self._generate_causal_mask(3 * L).to(device)

        # === TRANSFORMER === #
        h = self.transformer(h, mask=attention_mask)       # (B, 3L, H)

        # === PREDICTION FROM STATE POSITIONS === #
        state_pos = torch.arange(L, device=device) * 3 + 1
        h_states = h[:, state_pos, :]                      # (B, L, H)

        logits = self.predict_item(h_states)               # (B, L, num_items)
        return logits

    def _generate_causal_mask(self, seq_len):
        """
        Causal mask that respects:
        - All previous timesteps can attend
        - Within a timestep: RTG→STATE→ACTION order
        """
        mask = torch.full((seq_len, seq_len), float('-inf'))

        for i in range(seq_len):
            current_step = i // 3
            current_type = i % 3

            for j in range(i + 1):
                prev_step = j // 3
                prev_type = j % 3

                if prev_step < current_step:
                    mask[i, j] = 0
                elif prev_step == current_step and prev_type <= current_type:
                    mask[i, j] = 0

        return mask
