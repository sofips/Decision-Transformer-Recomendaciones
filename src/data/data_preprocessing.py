#-------------------------------------------------------------
# Pre-procesamiento de datos para usar Decision Transformers
#--------------------------------------------------------------
import numpy as np


def create_dt_dataset(df_train, test=True, print_stats=True):
    '''
    Pre-procesamiento de datos para entrenar un Decision Transformer. 

    Parametros:
        df_train (pd.DataFrame): DataFrame con las columnas 'user_id', 'user_group', 'items', 'ratings'.
    
    Returns:
        trajectories (list): Lista de diccionarios, cada uno representando una trayectoria de usuario. 
        Cada diccionario contiene:
            - 'items': array de numpy con los IDs de los items (películas) vistos por el usuario.
            - 'ratings': array de numpy con las calificaciones dadas por el usuario. Matchea con 'items'.
            - 'returns_to_go': array de numpy con los retornos futuros esperados.
            - 'timesteps': pasos temporales.
            - 'user_group': cluster o grupo (0-7) al que pertenece el usuario.
    '''

    trajectories = []

    # para cada usuario en el dataset
    for user_id, user_data in df_train.iterrows():
        items = user_data['items']
        ratings = user_data['ratings']
        user_group = user_data['user_group']

        # calcular returns-to-go (RTG)
        returns_to_go = np.zeros_like(ratings)

        returns_to_go[-1] = ratings[-1]

        for t in reversed(range(len(ratings)-1)):
            returns_to_go[t] = ratings[t] + returns_to_go[t + 1]

        timesteps = np.arange(len(items), dtype=np.int32)

        # crear entrada de la trayectoria
        trajectory = {
            'items': items,
            'ratings': ratings,
            'returns_to_go': returns_to_go,
            'timesteps': timesteps,
            'user_group': user_group
        }

        if test:
            # Validar que las longitudes coincidan
            if not (len(items) == len(ratings) == len(returns_to_go) == len(timesteps)):
                raise ValueError(f"Las longitudes de los arrays no coinciden para el usuario {user_id}.")
            # Chequear que la return-to-go en el paso 0 sea la suma total de ratings
            if returns_to_go[0] != np.sum(ratings):
                raise ValueError(f"El return-to-go inicial no coincide con la suma de ratings para el usuario {user_id}.")
            # Chequear que la return-to-go en el último paso sea igual al último rating
            if returns_to_go[-1] != ratings[-1]:
                raise ValueError(f"El return-to-go final no coincide con el último rating para el usuario {user_id}.")
        trajectories.append(trajectory)

    if test:
        if len(df_train['user_id']) != len(trajectories):
            raise ValueError("El número de usuarios no coincide con el número de trayectorias generadas.")

    if print_stats:
        print(f"Número de trayectorias generadas: {len(trajectories)}")
        lengths = [len(traj['items']) for traj in trajectories]
        print(f"Longitud media de las trayectorias: {np.mean(lengths):.2f}")
        print(f"Longitud máxima de las trayectorias: {np.max(lengths)}")
        print(f"Longitud mínima de las trayectorias: {np.min(lengths)}")

    return trajectories