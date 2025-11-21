import torch
import torch.nn as nn

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

        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        self.group_embedding = nn.Embedding(num_groups, hidden_dim)
        self.rtg_embedding = nn.Linear(1, hidden_dim)
        self.timestep_embedding = nn.Embedding(max_timesteps, hidden_dim)

        