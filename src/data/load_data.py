'''Módulo para cargar datasets de recomendación. Permite elegir entre
el dataset 'netflix' o 'goodreads'. Provee funciones para cargar los
conjuntos de training (load_training_data) y test (load_test_data), y
los centroides de grupos (load_group_centroids).
'''

import pandas as pd
import json

# ============================================
# CONFIGURACIÓN: Elegir dataset
# ============================================
DATASET = 'netflix'    # O 'goodreads'
NUM_ITEMS = 752 if DATASET == 'netflix' else 472

def load_training_data(dataset='netflix', path = '/data/train/'):
    """
    Carga el dataset de training.
    
    Args:
        dataset: 'netflix' o 'goodreads'
    
    Returns:
        df: pandas DataFrame con columnas [user_id, user_group, items, ratings]
    """
    
    path = f'{path}{dataset}8_train.df'
    df = pd.read_pickle(path)
    
    return df
    

def load_test_data(dataset='netflix', path = '/data/test_users/'):
    """
    Carga el dataset de test (cold-start users).
    
    Args:
        dataset: 'netflix' o 'goodreads'
    
    Returns:
        test_users: lista de diccionarios con keys [group, items, ratings]
    """
    path = f'{path}{dataset}8_test.json'
    
    with open(path, 'r') as f:
         return json.load(f)
    

def load_group_centroids(dataset='netflix', path = '/data/groups/'):
    """
    Carga centroides de grupos (OPCIONAL).
    
    Args:
        dataset: 'netflix' o 'goodreads'
    
    Returns:
        mu: DataFrame de 8xNUM_ITEMS con ratings promedio por grupo
    """
    path = f'{path}mu_{dataset}8.csv'
    
    mu = pd.read_csv(path, header=None)
    
    return mu
    