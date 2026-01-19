import random
import numpy as np
from collections import Counter, defaultdict
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, completeness_score
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score
from scipy.stats import mode

from src.utils import *
from config.parameters import *

def calcular_precision_contencion_clusters(y_val, clusters_val, y_train, clusters_train):
    """
    Calcula la precisión basada en si las clases reales están contenidas en los clusters asignados,
    manteniendo el estilo original del código.
    
    Parámetros:
        y_val: Etiquetas reales de validación
        clusters_val: Clusters asignados en validación 
        y_train: Etiquetas reales de entrenamiento
        clusters_train: Clusters asignados en entrenamiento
        
    Retorna:
        Precisión de contención (float)
    """
    # Mapeo de clusters a clases (basado en entrenamiento)
    mapa_cluster_a_clases = {}
    for cluster in np.unique(clusters_train):
        etiquetas_en_cluster = y_train[clusters_train == cluster]
        mapa_cluster_a_clases[cluster] = set(etiquetas_en_cluster)

    # Conteo de aciertos
    aciertos = 0
    for i in range(len(y_val)):
        cluster_actual = clusters_val[i]
        clase_esperada = y_val[i]
        if clase_esperada in mapa_cluster_a_clases.get(cluster_actual, set()):
            aciertos += 1

    return aciertos / len(y_val)

def evaluate_model(model, name,
                   X_train, y_train,
                   X_val, y_val,
                   val_df,
                   extended=False):
    """Evaluate a clustering model and display metrics."""

    # ----------------------
    # Helper functions
    # ----------------------
    def _cluster_info(labels):
        counts = Counter(labels)
        clusters = np.array(sorted(counts.keys(), key=lambda x: (x == -1, x)))
        sizes = np.array([counts[c] for c in clusters])
        return clusters, sizes, counts

    def _weighted_purity(labels, y_true):
        N = len(labels)
        df = pd.DataFrame({'cluster': labels, 'y': y_true})
        agg = df.groupby('cluster')['y'].value_counts()
        max_per_cluster = agg.groupby(level=0).max()
        return float(max_per_cluster.sum() / N)

    def _containment_score(val_clusters, y_val, train_clusters, y_train):
        df_train = pd.DataFrame({'cluster': train_clusters, 'y': y_train})
        cluster_sets = df_train.groupby('cluster')['y'].apply(lambda s: set(s))
        hits = [1 if c in cluster_sets and y in cluster_sets[c] else 0
                for c, y in zip(val_clusters, y_val)]
        return float(np.mean(hits))

    def _size_balance_score(sizes):
        if len(sizes) == 0:
            return 0.0
        mean = sizes.mean()
        if mean == 0:
            return 0.0
        cv = sizes.std(ddof=0) / mean
        return 1.0 / (1.0 + cv)

    # ----------------------
    # TRAINING
    # ----------------------
    if hasattr(model, 'fit_predict'):
        train_clusters = model.fit_predict(X_train)
    else:
        model.fit(X_train)
        train_clusters = model.predict(X_train)

    # ----------------------
    # MAP CLUSTER → LABEL
    # ----------------------
    cluster_to_label = {}
    for cluster in np.unique(train_clusters):
        labels = y_train[train_clusters == cluster]
        if len(labels) > 0:
            cluster_to_label[cluster] = mode(labels, keepdims=False).mode

    # ----------------------
    # VALIDATION
    # ----------------------
    val_clusters = (model.predict(X_val)
                    if hasattr(model, 'predict')
                    else model.fit_predict(X_val))
    y_val_pred = np.array([cluster_to_label.get(c, -1) for c in val_clusters])

    # ----------------------
    # CLASSIC METRICS
    # ----------------------
    metrics = {
        'Accuracy': accuracy_score(y_val, y_val_pred),
        'Completeness': completeness_score(y_val, val_clusters),
        'ARI': adjusted_rand_score(y_val, val_clusters),
        'NMI': normalized_mutual_info_score(y_val, val_clusters),
        'Homogeneity': homogeneity_score(y_val, val_clusters)
    }

    # ----------------------
    # STRUCTURAL METRICS
    # ----------------------
    clusters, sizes, counts = _cluster_info(train_clusters)
    N_train = len(train_clusters)

    purity = _weighted_purity(train_clusters, y_train)
    containment = _containment_score(val_clusters, y_val, train_clusters, y_train)
    size_balance = _size_balance_score(sizes)

    tiny_threshold = max(1, int(DEFAULT_MIN_CLUSTER_RATIO * N_train))
    n_tiny = np.sum(sizes < tiny_threshold)
    frac_tiny = n_tiny / max(1, len(sizes))
    tiny_cluster_penalty = 1.0 - frac_tiny

    noise_frac = counts.get(-1, 0) / N_train
    noise_penalty = 1.0 - noise_frac

    # ----------------------
    # COMPOSITE SCORE
    # ----------------------
    components = {
        'containment': containment,
        'purity': purity,
        'size_balance': size_balance,
        'tiny_cluster_penalty': tiny_cluster_penalty,
        'noise_penalty': noise_penalty
    }

    composite_score = 0.0
    for k, w in DEFAULT_COMPOSITE_WEIGHTS.items():
        v = components.get(k, 0.0)
        if not np.isfinite(v):
            v = 0.0
        composite_score += w * min(max(v, 0.0), 1.0)

    # ======================
    # SALIDA UNIFICADA
    # ======================
    print(f"\nResults for model: {name}")

    print("\nClassic metrics:")
    for metric, value in metrics.items():
        print(f"{metric:<20}: {value:.4f}")

    print("\nStructural metrics:")
    print(f"{'Containment':<20}: {containment:.4f}")
    print(f"{'Purity':<20}: {purity:.4f}")
    print(f"{'Size balance':<20}: {size_balance:.4f}")
    print(f"{'Tiny cluster frac':<20}: {frac_tiny:.4f}")
    print(f"{'Noise frac (-1)':<20}: {noise_frac:.4f}")

    print(f"\nComposite score: {composite_score:.4f}")
    print(f"Cluster sizes: {dict(counts)}")

    # ======================
    # ANÁLISIS EXTENDIDO
    # ======================

    if extended:
        show_extended_analysis(y_train, train_clusters, y_val, val_clusters, y_val_pred, val_df)

def encontrar_mejor_random_state(X, n_clusters=8, max_seeds=20):
    mejores_resultados = []
    
    for seed in range(max_seeds):
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
        kmeans.fit(X)
        inercia = kmeans.inertia_
        mejores_resultados.append((seed, inercia))
    
    # Ordenar por mejor inercia (menor es mejor)
    mejores_resultados.sort(key=lambda x: x[1])
    
    print("Top 5 mejores semillas:")
    for seed, inercia in mejores_resultados[:5]:
        print(f"random_state={seed}: Inercia = {inercia:.4f}")
    
    return mejores_resultados[0][0]  # Retorna la mejor semilla

def split_train_test(entries, train_ratio=2/3):
    entries_list = list(entries.items())
    by_class = defaultdict(list)

    for key, value in entries_list:
        by_class[value[0]].append((key, value))

    train, remaining = [], []

    for elements in by_class.values():
        random.shuffle(elements)
        train.append(elements[0])      # at least one per class
        remaining.extend(elements[1:])

    random.shuffle(remaining)

    desired_train_size = int(len(entries_list) * train_ratio)
    missing = desired_train_size - len(train)

    train.extend(remaining[:missing])
    test = remaining[missing:]

    return train, test

def show_extended_analysis(y_true, train_clusters, y_val, val_clusters, y_val_pred, val_df):
    """Displays detailed cluster distribution and validation summary."""
    from collections import Counter
    from scipy.stats import mode

    # ----------------------
    # TRAINING CLUSTER DISTRIBUTION
    # ----------------------
    cluster_dict = {}
    for cluster in np.unique(train_clusters):
        labels = y_true[train_clusters == cluster]
        cluster_dict[cluster] = dict(Counter(labels))

    sorted_clusters = sorted(cluster_dict.items(), key=lambda x: x[0])
    print(f"\nTraining distribution {len(sorted_clusters)} clusters:")
    for cluster, dist in sorted_clusters:
        total = sum(dist.values())
        dist_ordered = dict(sorted({int(k): v for k, v in dist.items()}.items()))
        print(f"Cluster {cluster}: {total} samples | Classes: {len(dist)} | Distribution: {dist_ordered}")

    # ----------------------
    # MAP LABEL → MOST COMMON CLUSTER
    # ----------------------
    mapping = {}
    for label in np.unique(y_true):
        clusters_for_label = train_clusters[y_true == label]
        if len(clusters_for_label) > 0:
            mapping[label] = mode(clusters_for_label, keepdims=False).mode

    # ----------------------
    # WRITE VALIDATION SUMMARY
    # ----------------------
    '''
    with open(EXTENDED_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(f"\nValidation samples (Expected, Predicted, Expected cluster, Predicted cluster):\n")
        f.write("Cat\tCon\tDis\t#Ins\tDesb\tCla\tBal\t|\tBal Exp\tBal Pred\tClu Exp\tClu Pred\n")

        higher = 0
        lower = 0
        equal = 0

        for i in range(min(len(y_val), EXTENDED_MAX_ROWS)):
            expected = y_val[i]
            predicted = y_val_pred[i]
            cluster_expected = mapping.get(expected, "-")
            cluster_predicted = val_clusters[i]

            len_exp = len(cluster_dict.get(cluster_expected, {}))
            len_pred = len(cluster_dict.get(cluster_predicted, {}))
            row_data = "\t".join(str(x) for x in val_df.loc[i].values)

            # Write row to file
            f.write(f"{row_data}\t|\t{expected:<3}\t{predicted:<3}\t{cluster_expected}-{len_exp}\t{cluster_predicted}-{len_pred}\n")

            if len_exp > len_pred:
                higher += 1
            elif len_exp < len_pred:
                lower += 1
            else:
                equal += 1

        f.write(f"Higher: {higher}; Lower: {lower}; Equal: {equal}\n")

    print("Results saved to validation file.")
    '''
