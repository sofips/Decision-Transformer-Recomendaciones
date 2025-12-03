import numpy as np
import torch as T
from math import log2


'''
Funciones de evaluación de modelos.

- evaluate_popularity: evalúa el modelo de Popularidad.
- evaluate_model: evalúa un Decision Transformer.
'''

def evaluate_popularity(pop_model, test_data, k_list=[5, 10, 20], context_len=20):
    """
    Evalúa el baseline de Popularidad en el test set.
    
    Usa solo el ranking global de popularidad
    (pop_model.popular_items) y el history de cada usuario para filtrar ítems vistos.
    
    Devuelve HR@K, NDCG@K y MRR promediados.
    """
    metrics = {f'HR@{k}': [] for k in k_list}
    metrics.update({f'NDCG@{k}': [] for k in k_list})
    metrics['MRR'] = []

    # IDCG para NDCG (target en posición 1)
    idcg = 1.0 / np.log2(2)  # = 1.0

    for user in test_data:
        items = user['items'] 

        for t in range(context_len, len(items)):
            history_items = items[t-context_len:t]
            target_item = items[t]

            seen = set(history_items)

            # Buscar la posición (rank) del target en el ranking de popularidad
            # filtrando los items ya vistos
            rank = None
            current_rank = 0

            for item in pop_model.popular_items:
                if item in seen:
                    continue
                current_rank += 1
                if item == target_item:
                    rank = current_rank
                    break

            # Si no se encuentra el target en el ranking filtrado,
            # lo consideramos como fallo (métricas = 0 para ese caso).
            if rank is None:
                for k in k_list:
                    metrics[f'HR@{k}'].append(0.0)
                    metrics[f'NDCG@{k}'].append(0.0)
                metrics['MRR'].append(0.0)
                continue

            # HR@K: 1 si rank <= K, 0 si no
            for k in k_list:
                hit = 1.0 if rank <= k else 0.0
                metrics[f'HR@{k}'].append(hit)

                # NDCG@K: 1/log2(rank+1) si rank <= K, 0 si no
                if rank <= k:
                    dcg = 1.0 / np.log2(rank + 1)
                    ndcg = dcg / idcg
                else:
                    ndcg = 0.0
                metrics[f'NDCG@{k}'].append(ndcg)

            # MRR: 1/rank
            rr = 1.0 / rank
            metrics['MRR'].append(rr)

    # Promediar métricas
    return {key: float(np.mean(vals)) for key, vals in metrics.items()}


@T.no_grad()

def evaluate_model_batched(
    model,
    test_data,
    device,
    target_return=None,
    k_list=(5, 10, 20),
    context_len=20,
    eval_batch_size=1024
):
    """
    Evaluación batcheada y GPU-native para usuarios con longitudes distintas.


    Reimplementamos la evaluación para ejecutar en gpu
    Args:
        model: Decision Transformer que acepta (states, actions, rtg, timesteps, groups)
        test_data: lista de dicts {'group': int, 'items': List[int], 'ratings': List[float]}
        device: T.device
        target_return: float or None (si None usa suma de ratings de la ventana)
        k_list: tupla/lista de K para métricas
        context_len: longitud de la historia a usar (ventana)
        eval_batch_size: tamaño de batch para la inferencia (ajustar por memoria)
    Returns:
        dict con HR@k, NDCG@k, MRR (floats)
    """

    model.eval()

    # 1) Construir dataset de ventanas (en CPU) - cada muestra = una historia de length context_len + target
    states_list = []
    actions_list = []
    rtg_list = []
    groups_list = []
    targets_list = []

    for user in test_data:
        group = int(user['group'])
        items = user['items']
        ratings = user['ratings']

        L = len(items)
        if L <= context_len:
            continue  # no hay ventana válida

        # generar ventanas
        for t in range(context_len, L):
            hist_items = items[t-context_len:t]               # length = context_len
            hist_ratings = ratings[t-context_len:t]

            rtg_val = (sum(hist_ratings) if target_return is None else float(target_return))

            states_list.append(hist_items)
            actions_list.append(hist_items)
            rtg_list.append([rtg_val] * context_len)
            groups_list.append(group)
            targets_list.append(items[t])

    # Si no hay muestras válidas
    if len(states_list) == 0:
        return {f'HR@{k}': 0.0 for k in k_list} | {f'NDCG@{k}': 0.0 for k in k_list} | {'MRR': 0.0}

    # 2) Convertir todo a tensores y mover a device UNA VEZ (evita múltiples CPU->GPU)
    states = T.tensor(states_list, dtype=T.long, device=device)        # (N, context_len)
    actions = T.tensor(actions_list, dtype=T.long, device=device)      # (N, context_len)
    rtg = T.tensor(rtg_list, dtype=T.float32, device=device).unsqueeze(-1)  # (N, context_len, 1)
    groups = T.tensor(groups_list, dtype=T.long, device=device)        # (N,)
    targets = T.tensor(targets_list, dtype=T.long, device=device)      # (N,)

    N = states.size(0)
    num_ks = len(k_list)

    # Precompute timesteps (se puede broadcastear por batch)

    timesteps_single = T.arange(context_len, dtype=T.long, device=device).unsqueeze(0)  # (1, context_len)

    # 3) Acumuladores para métricas (mantener en GPU)
    hr_sums = T.zeros(num_ks, device=device, dtype=T.float64)   # usamos float64 para mayor estabilidad al acumular
    ndcg_sums = T.zeros(num_ks, device=device, dtype=T.float64)
    mrr_sum = T.tensor(0.0, device=device, dtype=T.float64)
    total = 0

    max_k = max(k_list)
    ranks = T.arange(1, max_k + 1, device=device, dtype=T.float32)  # (max_k,)
    discount = T.log2(ranks + 1.0)  # (max_k,)

    # 4) Inferencia en mini-batches
    for start in range(0, N, eval_batch_size):
        end = min(start + eval_batch_size, N)
        b = end - start

        s_batch = states[start:end]           # (b, context_len)
        a_batch = actions[start:end]          # (b, context_len)
        r_batch = rtg[start:end]              # (b, context_len, 1)
        g_batch = groups[start:end]           # (b,)
        t_batch = targets[start:end]          # (b,)

        # timesteps repeat para el batch
        ts_batch = timesteps_single.expand(b, -1)  # (b, context_len)

        # forward (asume salida logits (b, seq_len, num_items) )
        logits = model(s_batch, a_batch, r_batch, ts_batch, g_batch)  # (b, seq_len, num_items)
        preds = logits[:, -1, :]   # (b, num_items)  <-- scores para cada item

        # --- HR@K and NDCG@K ---
        # obtener top max_k indices y scores
        topk_vals, topk_idx = T.topk(preds, k=max_k, dim=1)  # (b, max_k)
        # targets comparacion
        # shape targets -> (b,1) for broadcasting
        eq = (topk_idx == t_batch.unsqueeze(1))  # (b, max_k) bool

        # Para cada K en k_list computar HR y NDCG
        for i, k in enumerate(k_list):
            eq_k = eq[:, :k]                                 # (b, k)
            hits = eq_k.any(dim=1).to(T.float32)         # (b,)
            hr_sums[i] += hits.sum().to(T.float64)

            # NDCG: relevance is 1 only where eq_k True, DCG = sum(relevance / log2(rank+1))
            # discount[:k] -> (k,)
            relevance = eq_k.to(T.float32)              # (b, k)
            dcg = (relevance / discount[:k]).sum(dim=1)     # (b,)
            ndcg_sums[i] += dcg.sum().to(T.float64)

        # --- MRR ---
        # Fast rank calculation without sorting fully:
        # rank_i = 1 + sum_j (preds_ij > preds_i,target)
        # target scores
        idx = T.arange(preds.size(0), device=device)
        target_scores = preds[idx, t_batch]                 # (b,)
        # count how many items have strictly greater score than target score
        better_count = (preds > target_scores.unsqueeze(1)).sum(dim=1).to(T.float32)  # (b,)
        ranks_tensor = better_count + 1.0
        rr = (1.0 / ranks_tensor).to(T.float64)        # (b,)
        mrr_sum += rr.sum()

        total += b

    # 5) Calcular promedios finales (mover a CPU 1 vez con .item())
    result = {}
    total = float(total)  # convertir a float Python

    for i, k in enumerate(k_list):
        result[f'HR@{k}'] = float((hr_sums[i] / total).item())
        result[f'NDCG@{k}'] = float((ndcg_sums[i] / total).item())

    result['MRR'] = float((mrr_sum / total).item())

    return result
