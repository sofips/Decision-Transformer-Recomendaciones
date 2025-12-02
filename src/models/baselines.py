import numpy as np

'''
Popularity Recommender System.
Este modelo recomienda los ítems en base a su popularidad.

'''

class PopularityRecommender:
    def __init__(self):
        self.item_counts = None
        self.popular_items = None
    
    def fit(self, train_data):
        # Contar frecuencia de cada item
        all_items = np.concatenate([traj['items'] for traj in train_data])
        # Para Netflix, hay 752 ítems; dejamos 752 como en la consigna
        self.item_counts = np.bincount(all_items, minlength=752)
        # Ordenar por frecuencia (más popular primero)
        self.popular_items = np.argsort(self.item_counts)[::-1]
    
    def recommend(self, user_history, k=10):
        # Recomendar los k items más populares que no estén en history
        recommendations = []
        for item in self.popular_items:
            if item not in user_history:
                recommendations.append(item)
            if len(recommendations) == k:
                break
        return recommendations
    