# %%
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import datetime

from dt_reference import DecisionTransformerReference
from dataset_reference import RecommendationDatasetReference
from train_reference import train_decision_transformer_reference

import numpy as np
import torch as T
from torch.utils.data import random_split, DataLoader

# Set up device
device = T.device('cuda' if T.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def set_seed(seed):
    np.random.seed(seed)
    T.manual_seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed_all(seed)
set_seed(42)

# %% [markdown]
# # Entrenamiento de DT4REC

# %% [markdown]
# En este notebook se implementa un pipeline de entrenamiento de un Transformer de Decisiones (DT) para predecir recomendaciones partiendo de un dataset de Netflix.

# %% [markdown]
# ### Carga de trayectorias
# 
# Se cargan las trayectorias previamente preprocesadas. 
# 
# Se tienen dos archivos de trayectoria uno con las recompensas normalizadas ('data/processed/normalized_trajectories_train.pkl') y otro con las recompensas sin normalizar (data/processed/trajectories_train.pkl). 

# %%
import pickle

load_normalized_trayectories = True

if load_normalized_trayectories:
    with open('../data/processed/normalized_trajectories_train.pkl', 'rb') as f:
        trajectories = pickle.load(f)
else:
    with open('../data/processed/trajectories_train.pkl', 'rb') as f:
        trajectories = pickle.load(f)

# %% [markdown]
# ### Configuración de Híper-parámetros.
# 
# Se realizaron algunas pruebas sobre distintos conjuntos de híper-parámetros. Finalmente, estos fueron los que proveyeron mejores resultados:

# %%
# Hyperparameters
num_items = 752
num_groups = 8
hidden_dim = 128
n_layers = 3
n_heads = 4
context_length = 20
max_timesteps = 200
dropout = 0.1
batch_size = 64
num_epochs = 2000
learning_rate = 0.0001

# %% [markdown]
# ### Inicialización del modelo
# 
# Con los híper-parámetros elegidos, se instancia la clase `DecisionTransformer` que contiene el modelo a entrenar

# %%
# Inicialización del modelo
model = DecisionTransformerReference(
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


# %% [markdown]
# ### Preparación de datos para entrenamiento
# 
# Usando `RecommendationDataset` construimos el conjunto de datos en el formato admitido por el Decision Transformer. Como resultado, se obtiene un diccionario con tensores para 'states', 'actions', 'rtg', 'timesteps', 'groups' y 'targets'.
# 
# En esta instancia separamos un subconjunto para validación. Seteamos el tamaño en un 20% del dataset completo.
# 

# %%
dataset = RecommendationDatasetReference(trajectories=trajectories, context_length=context_length)
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

print("\nDataset details:")
print(f"Dataset train size: {len(train_dataset)}")
print(f"Batches per epoch (train): {len(train_loader)}")
print(f"Dataset val size: {len(val_dataset)}")
print(f"Batches per epoch (val): {len(val_loader)}")

# Para testear los shapes de los batches
# sample_batch = next(iter(train_loader))
# print(f"\nSample batch shapes:")
# for key, val in sample_batch.items():
#     print(f"  {key}: {val.shape}")



# %%
optimizer = T.optim.Adam(model.parameters(), lr=learning_rate)


model, history = train_decision_transformer_reference(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=num_epochs,
    checkpoint_dir="checkpoints"
)


# %%


#Save the trained model
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
T.save(model.state_dict(), f'trained_model_{date}.pt')
print(f"Model saved to 'trained_model_{date}.pt'")

#Save the training history
with open(f'training_history_{date}.pkl', 'wb') as f:
    pickle.dump(history, f)
print(f"Training history saved to 'training_history_{date}.pkl'")