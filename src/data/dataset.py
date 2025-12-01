import numpy as np
import torch
from torch.utils.data import Dataset


class RecommendationDataset(Dataset):
    def __init__(self, trajectories, context_length=20):
        """
        Dataset para Decision Transformer en recomendaciones.
        trajectories: list of dicts with keys:
            'items' (list/np.array of ints in [0..num_items-1]),
            'returns_to_go' (array), 'timesteps' (array), 'user_group' (int)
        context_length: max length L
        NOTE: dataset shifts item ids by +1 so that 0 is padding.
        """
        self.trajectories = trajectories
        self.context_length = context_length
        
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        
        # Extraer datos de la trayectoria
        items = np.asarray(traj['items'], dtype=np.int64)  # 0..num_items-1
        rtg = np.asarray(traj['returns_to_go'], dtype=np.float32)
        timesteps = np.asarray(traj['timesteps'], dtype=np.int64)
        group = int(traj['user_group'])
        
        seq_len = min(len(items), self.context_length)
        
        # Seleccionar ventana aleatoria si la secuencia es larga
        if len(items) > self.context_length:
            start_idx = np.random.randint(0, len(items) - self.context_length + 1)
        else:
            start_idx = 0
        
        end_idx = start_idx + seq_len
        
        # States: items[start:end]
        states = items[start_idx:end_idx].copy()
        
        # Actions: next item (t+1) else 0 (padding)
        actions = np.full(seq_len, 0, dtype=np.int64)   # 0 = padding BEFORE shift
        if seq_len > 1:
            actions[:-1] = items[start_idx+1:end_idx]
        
        targets = actions.copy()
        
        # Returns-to-go (reshape to (L,1))
        rtg_seq = rtg[start_idx:end_idx].reshape(-1, 1)
        
        # Timesteps relative to window start
        time_seq = timesteps[start_idx:end_idx] - start_idx
        
        # SHIFT items by +1 so that 0 is reserved for padding
        return {
            'states': torch.tensor(states, dtype=torch.long) + 1,     # shift
            'actions': torch.tensor(actions, dtype=torch.long) + 1,   # shift
            'rtg': torch.tensor(rtg_seq, dtype=torch.float32),
            'timesteps': torch.tensor(time_seq, dtype=torch.long),
            'groups': torch.tensor(group, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long) + 1    # shift
        }
