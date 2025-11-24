import numpy as np
# ===================================================================
# CÓDIGO DE REFERENCIA - Muestra el formato esperado
# Los grupos deben implementar esto (o similar) en data_preprocessing.py
# ===================================================================

def create_dt_dataset(df_train):
    """
    Convierte el dataset raw a formato Decision Transformer.
    
    Args:
        df_train: DataFrame con columnas [user_id, user_group, items, ratings]
    
    Returns:
        trajectories: List[Dict] donde cada dict representa una trayectoria
                      con las siguientes keys:
            - 'items': numpy array de item IDs (secuencia completa)
            - 'ratings': numpy array de ratings (secuencia completa)
            - 'returns_to_go': numpy array con suma de rewards futuros desde cada timestep
            - 'timesteps': numpy array con índices temporales [0, 1, 2, ..., T-1]
            - 'user_group': int con el grupo del usuario (0-7)
    
    Ejemplo de salida:
        [
          {
            'items': array([472, 97, 122, ...]),      # 112 elementos
            'ratings': array([4., 3., 4., ...]),       # 112 elementos
            'returns_to_go': array([450., 446., ...]), # 112 elementos (suma acumulada hacia adelante)
            'timesteps': array([0, 1, 2, ...]),        # 112 elementos
            'user_group': 2
          },
          ... # 16,000 trayectorias total
        ]
    """
    trajectories = []
    
    # Iterar sobre cada usuario
    for idx, row in df_train.iterrows():
        items = row['items']        # numpy array de item IDs
        ratings = row['ratings']    # numpy array de ratings
        group = row['user_group']   # int (0-7)
        
        # === PASO CLAVE: Calcular returns-to-go ===
        # R̂_t = suma de rewards desde t hasta el final
        # R̂_t = r_t + r_{t+1} + ... + r_T
        
        returns = np.zeros(len(ratings))
        
        # Último timestep: R̂_T = r_T
        returns[-1] = ratings[-1]
        
        # Iterar hacia atrás: R̂_t = r_t + R̂_{t+1}
        for t in range(len(ratings)-2, -1, -1):
            returns[t] = ratings[t] + returns[t+1]
        
        # Ejemplo:
        # ratings =  [4, 3, 5, 2, 1]
        # returns = [15, 11, 8, 3, 1]  # 15=4+3+5+2+1, 11=3+5+2+1, etc.
        
        # Crear diccionario con toda la información
        trajectory = {
            'items': items,                        # Secuencia de películas
            'ratings': ratings,                    # Ratings correspondientes
            'returns_to_go': returns,              # R̂ para cada timestep
            'timesteps': np.arange(len(items)),    # [0, 1, 2, ..., T-1]
            'user_group': group                    # Cluster del usuario
        }
        
        trajectories.append(trajectory)
    
    return trajectories


# === EJEMPLO DE USO ===
# import pandas as pd
# df_train = pd.read_pickle('data/train/netflix8_train.df')
# trajectories = create_dt_dataset(df_train)
# print(f"Total trayectorias: {len(trajectories)}")  # 16,000
# print(f"Primera trayectoria keys: {trajectories[0].keys()}")
# print(f"Returns-to-go del usuario 0: {trajectories[0]['returns_to_go'][:10]}")