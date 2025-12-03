
## adaptado del notebook para correr en terminal

import sys
import os
sys.path.insert(0, os.path.abspath('..'))


from src.models.decision_transformer import DecisionTransformer
from src.data.dataset import RecommendationDataset
from src.training.trainer import train_decision_transformer
import pickle
import numpy as np
import torch as T
from torch.utils.data import random_split, DataLoader
import datetime

# Set up device
device = T.device('cuda' if T.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def set_seed(seed):
    np.random.seed(seed)
    T.manual_seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed_all(seed)
set_seed(42)


load_normalized_trayectories = True

if load_normalized_trayectories:
    with open('../data/processed/normalized_trajectories_train.pkl', 'rb') as f:
        trajectories = pickle.load(f)
else:
    with open('../data/processed/trajectories_train.pkl', 'rb') as f:
        trajectories = pickle.load(f)

# ### Configuración de Híper-parámetros.
#
# Se realizaron algunas pruebas sobre distintos conjuntos de híper-parámetros. 
# Finalmente, estos fueron los que proveyeron mejores resultados:

# Hyperparameters
num_items = 752
num_groups = 8
hidden_dim = 256
n_layers = 2
n_heads = 4
context_length = 25
max_timesteps = 200
dropout = 0.1
batch_size = 64
num_epochs = 2000
learning_rate = 0.0001


# ### Inicialización del modelo
#
# Con los híper-parámetros elegidos, se instancia la clase `DecisionTransformer` que contiene el modelo a entrenar

# Inicialización del modelo
model = DecisionTransformer(
    num_items=num_items,
    num_groups=num_groups,
    hidden_dim=hidden_dim,
    n_layers=n_layers,
    n_heads=n_heads,
    context_length=context_length,
    max_timestep=max_timesteps,
    dropout=dropout
).to(device)

# Número de parámetros del modelo
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ### Preparación de datos para entrenamiento
#
# Usando `RecommendationDataset` construimos el conjunto de datos en el formato admitido por el Decision Transformer. Como resultado, se obtiene un diccionario con tensores para 'states', 'actions', 'rtg', 'timesteps', 'groups' y 'targets'.
#
# En esta instancia separamos un subconjunto para validación. Seteamos el tamaño en un 20% del dataset completo.
#

dataset = RecommendationDataset(trajectories=trajectories, context_length=context_length)
print(dataset.trajectories[0])

print(f"Number of training trajectories: {len(trajectories)}")
print(dataset)

val_ratio = 0.1 # porcentaje de datos para validación
train_ratio = 1 - val_ratio

n_total = len(dataset)
n_train = int(n_total * train_ratio)
n_val = n_total - n_train

train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,        # siempre validamos sobre la misma secuencia
    num_workers=0
)

optimizer = T.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)

train = True

model, history = train_decision_transformer(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=1000,
    checkpoint_dir="checkpoints"
)


# Guardamos el modelo entrenado
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
T.save(model.state_dict(), f'trained_model_{date}.pt')
print(f"Model saved to 'trained_model_{date}.pt'")


# Guardamos el historial de entrenamiento
with open(f'training_history_{date}.pkl', 'wb') as f:
    pickle.dump(history, f)
print(f"Training history saved to 'training_history_{date}.pkl'")


