import numpy as np
import torch as T
from torch.utils.data import Dataset

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
            'states': T.tensor(states, dtype=T.long),
            'actions': T.tensor(actions, dtype=T.long),
            'rtg': T.tensor(rtg_seq, dtype=T.float32),
            'timesteps': T.tensor(time_seq, dtype=T.long),
            'groups': T.tensor(group, dtype=T.long),
            'targets': T.tensor(targets, dtype=T.long)
        }