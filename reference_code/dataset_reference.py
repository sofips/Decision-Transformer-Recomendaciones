from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class RecommendationDatasetReference(Dataset):
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
