from collections import Counter
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MeanShift, KMeans

from src.utils import get_labels
# =========================================================
# Switch de balanceadores
# =========================================================
def balance_dataset(method, X, y, config):
    if method == "debohid":
        return debohid_oversample(
            X, y,
            target_size=config.get("target_size"),
            F=config.get("F", 0.5),
            CR=config.get("CR", 0.7),
            generations=config.get("generations", 100)
        )
    elif method == "ngsmote":
        return ngsmote(X, y, **config)
    elif method == "hsvm":
        return hsvm_oversample(X, y, **config)
    elif method == "sldmax":
        return sldmax_oversample(X, y, **config)
    elif method == "rutsvm":
        return rutsvm_oversample(X, y, **config)
    elif method == "balancecascade":
        return balancecascade_undersample(X, y, **config)
    elif method == "bsmote":
        return bsmote_oversample(X, y, **config)
    elif method == "cbwkelm":
        return cbwkelm_balance(X, y, **config)
    elif method == "msoss":
        return msoss_oversample(X, y, **config)
    elif method == "ssmote":
        return ssmote(X, y, **config)
    elif method == "srsb":
        return srsb(X, y, **config)
    elif method == "iws_smote":
        return iws_smote(X, y, **config)
    if method == "sgo":
        return sgo(X, y, **config)
    if method == "smoteenn":
        return smote_enn(X, y, **config)
    if method == "ubkelm":
        return ubkelm_mv(X, y, **config)
    if method == "ufidsf":
        return ufidsf_chebyshev(X, y, **config)
    elif method in ["none", "original"]:
        # No se hace nada, se devuelve el dataset tal cual
        return X, y
    else:
        raise ValueError(f"Método {method} no implementado")

# =========================================================
# Implementación DEBOHID
# =========================================================
def debohid_oversample(X, y, target_size=None, F=0.5, CR=0.7, generations=100):
    X_min = X[y == 1]
    X_maj = X[y == 0]
    n_min, n_maj = len(X_min), len(X_maj)
    if target_size is None:
        target_size = n_maj

    population = [X_min[random.randrange(n_min)].copy() for _ in range(target_size)]

    for _ in range(generations):
        for i in range(target_size):
            a, b, c = random.sample(range(target_size), 3)
            x_a, x_b, x_c = population[a], population[b], population[c]
            mutant = x_a + F * (x_b - x_c)
            cross = np.array([
                mutant[j] if random.random() < CR else population[i][j]
                for j in range(X.shape[1])
            ], dtype=X.dtype)
            population[i] = cross

    synthetic = np.vstack(population)
    y_synth = np.array([1]*len(synthetic))

    X_new = np.vstack((X, synthetic))
    y_new = np.concatenate((y, y_synth))
    return X_new, y_new

def ngsmote(X, y, target_size, k_neighbors=5):
    """
    Genera muestras sintéticas para la clase minoritaria utilizando una distribución normal.

    Parámetros:
        X (numpy.ndarray): Características del conjunto de datos.
        y (numpy.ndarray): Etiquetas del conjunto de datos.
        target_size (int): Número total de muestras deseadas para la clase minoritaria.
        k_neighbors (int): Número de vecinos más cercanos a considerar para la generación de muestras.

    Retorna:
        X_res (numpy.ndarray): Características del conjunto de datos balanceado.
        y_res (numpy.ndarray): Etiquetas del conjunto de datos balanceado.
    """
    counts = Counter(y)
    minority_class_label = min(counts, key=counts.get)
    minority_instances = X[y == minority_class_label]
    n_to_generate = target_size - len(minority_instances)

    if n_to_generate <= 0 or len(minority_instances) < 2:
        # nada que generar
        return X, y

    nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(minority_instances)))
    nn.fit(minority_instances)

    synthetic_samples = []
    for _ in range(n_to_generate):
        instance = minority_instances[np.random.choice(len(minority_instances))]
        _, indices = nn.kneighbors([instance])
        neighbor = minority_instances[np.random.choice(indices[0])]
        diff = neighbor - instance
        noise = np.random.normal(loc=0, scale=1, size=diff.shape)
        synthetic_instance = instance + diff * np.random.uniform(0, 1) + noise
        synthetic_samples.append(synthetic_instance)

    synthetic_samples = np.array(synthetic_samples)
    X_res = np.vstack((X, synthetic_samples))
    y_res = np.hstack((y, np.full(n_to_generate, minority_class_label)))
    return X_res, y_res

def hsvm_oversample(X, y, target_size=None, k_neighbors=5, spread=0.5, C=1.0):
    X_min = X[y == 1]
    X_maj = X[y == 0]
    n_min, n_maj = len(X_min), len(X_maj)

    if target_size is None:
        target_size = n_maj

    if X.shape[1] < 1 or n_min < 2:
        return X, y

    # Entrenamos un SVM para encontrar vectores de soporte
    svm = SVC(kernel="linear", C=C)
    svm.fit(X, y)
    support_min = X_min[np.isin(np.where(y==1)[0], svm.support_)]
    if len(support_min) == 0:
        support_min = X_min

    # Vecinos cercanos de los vectores soporte
    n_neighbors_eff = min(k_neighbors, len(X_min))
    nn = NearestNeighbors(n_neighbors=n_neighbors_eff).fit(X_min)

    synthetic = []
    for _ in range(target_size):
        x = support_min[random.randrange(len(support_min))]
        nn_idx = nn.kneighbors([x], return_distance=False)[0]
        x_neighbor = X_min[random.choice(nn_idx)]
        diff = x_neighbor - x
        synthetic.append(x + spread * random.random() * diff)

    synthetic = np.array(synthetic)
    if synthetic.size == 0:  # por si no se generan muestras
        return X, y

    y_synth = np.array([1] * len(synthetic))
    X_new = np.vstack((X, synthetic))
    y_new = np.concatenate((y, y_synth))
    return X_new, y_new

def sldmax_oversample(X, y, target_size=None, k_neighbors=5, alpha=0.5, max_dist=1.0):
    X_min = X[y == 1]
    X_maj = X[y == 0]
    n_min, n_maj = len(X_min), len(X_maj)
    if target_size is None:
        target_size = n_maj

    nn = NearestNeighbors(n_neighbors=k_neighbors).fit(X)
    synthetic = []

    for _ in range(target_size):
        x = X_min[random.randrange(n_min)]
        nn_idx = nn.kneighbors([x], return_distance=False)[0]
        neighbors = X[nn_idx]
        labels = y[nn_idx]

        # safe-level: proporción de minoritarios en los vecinos
        safe_level = np.mean(labels == 1)

        # elige vecino minoritario
        min_neighbors = neighbors[labels == 1]
        if len(min_neighbors) == 0:
            continue
        x_neighbor = min_neighbors[random.randrange(len(min_neighbors))]

        # interpolación dependiente del safe level
        diff = x_neighbor - x
        lam = alpha * safe_level * random.random()
        lam = min(lam, max_dist)
        synthetic.append(x + lam * diff)

    synthetic = np.array(synthetic)
    y_synth = np.array([1] * len(synthetic))
    X_new = np.vstack((X, synthetic))
    y_new = np.concatenate((y, y_synth))
    return X_new, y_new

def rutsvm_oversample(X, y, target_size=None, n_iterations=10, C=1.0, spread=0.5):
    X_min = X[y == 1]
    X_maj = X[y == 0]
    n_min, n_maj = len(X_min), len(X_maj)
    if target_size is None:
        target_size = n_maj

    synthetic = []

    for _ in range(n_iterations):
        # Under-sample de la mayoría
        X_maj_res, y_maj_res = resample(X_maj, np.zeros(len(X_maj)), n_samples=n_min, random_state=None)
        X_train = np.vstack([X_min, X_maj_res])
        y_train = np.concatenate([np.ones(n_min), np.zeros(len(y_maj_res))])

        # Entrenar SVM
        svm = SVC(kernel="linear", C=C)
        svm.fit(X_train, y_train)

        # Tomar vectores de soporte minoritarios
        support_idx = [i for i in svm.support_ if y_train[i] == 1]
        support_min = X_train[support_idx]
        if len(support_min) == 0:
            support_min = X_min

        # Generar sintéticos
        for x in support_min:
            neighbor = X_min[random.randrange(len(X_min))]
            diff = neighbor - x
            synthetic.append(x + spread * random.random() * diff)
            if len(synthetic) >= target_size:
                break
        if len(synthetic) >= target_size:
            break

    synthetic = np.array(synthetic)
    y_synth = np.array([1] * len(synthetic))
    X_new = np.vstack((X, synthetic))
    y_new = np.concatenate((y, y_synth))
    return X_new, y_new

def balancecascade_undersample(X, y, target_size=None, n_estimators=10, n_iterations=5):
    X_min = X[y == 1]
    X_maj = X[y == 0]
    n_min, n_maj = len(X_min), len(X_maj)
    if target_size is None:
        target_size = n_min

    X_res, y_res = X_min.copy(), np.ones(n_min)

    maj_pool = X_maj.copy()
    maj_labels = np.zeros(len(maj_pool))

    for _ in range(n_iterations):
        if len(maj_pool) <= target_size:
            break

        # Under-sample mayoría al mismo tamaño que minoría
        X_maj_res, y_maj_res = resample(maj_pool, maj_labels, n_samples=n_min, random_state=None)
        X_train = np.vstack([X_min, X_maj_res])
        y_train = np.concatenate([np.ones(n_min), np.zeros(len(y_maj_res))])

        # Clasificador
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X_train, y_train)

        # Predicciones en mayoría
        y_pred = clf.predict(maj_pool)

        # Guardar mal clasificados (difíciles) para próxima iteración
        maj_pool = maj_pool[y_pred == 1]

        # Actualizar conjunto balanceado
        X_res = np.vstack([X_res, X_maj_res])
        y_res = np.concatenate([y_res, y_maj_res])

        if len(X_res) >= target_size * 2:
            break

    return X_res, y_res

def bsmote_oversample(X, y, target_size=None, k_neighbors=5, m_neighbors=10):
    X_min = X[y == 1]
    X_maj = X[y == 0]
    n_min, n_maj = len(X_min), len(X_maj)
    if target_size is None:
        target_size = n_maj

    nn = NearestNeighbors(n_neighbors=m_neighbors).fit(X)
    synthetic = []

    for _ in range(target_size):
        x = X_min[random.randrange(n_min)]
        nn_idx = nn.kneighbors([x], return_distance=False)[0]
        neighbors = X[nn_idx]
        labels = y[nn_idx]

        # vecino mayoritario cercano → candidato a frontera
        if np.any(labels == 0):
            min_neighbors = neighbors[labels == 1]
            if len(min_neighbors) > 0:
                x_neighbor = min_neighbors[random.randrange(len(min_neighbors))]
                diff = x_neighbor - x
                lam = random.random()
                synthetic.append(x + lam * diff)

    synthetic = np.array(synthetic)
    y_synth = np.array([1]*len(synthetic))
    X_new = np.vstack((X, synthetic))
    y_new = np.concatenate((y, y_synth))
    return X_new, y_new

def cbwkelm_balance(X, y, target_size=None, n_clusters=5, alpha=0.5, beta=1.0):
    X_min = X[y == 1]
    X_maj = X[y == 0]
    n_min, n_maj = len(X_min), len(X_maj)
    if target_size is None:
        target_size = n_maj

    # Clustering solo de minoritarios
    kmeans = KMeans(n_clusters=min(n_clusters, len(X_min)), n_init=5)
    labels = kmeans.fit_predict(X_min)

    synthetic = []
    for cluster_id in range(n_clusters):
        cluster_points = X_min[labels == cluster_id]
        if len(cluster_points) == 0:
            continue
        center = kmeans.cluster_centers_[cluster_id]
        for _ in range(target_size // n_clusters):
            x = cluster_points[random.randrange(len(cluster_points))]
            diff = center - x
            lam = alpha + beta * random.random()
            synthetic.append(x + lam * diff)

    synthetic = np.array(synthetic)
    y_synth = np.array([1]*len(synthetic))
    X_new = np.vstack((X, synthetic))
    y_new = np.concatenate((y, y_synth))
    return X_new, y_new

def msoss_oversample(X, y, target_size=None, bandwidth=1.0, cutoff=0.8, alpha=1.0):
    X_min = X[y == 1]
    X_maj = X[y == 0]
    n_min, n_maj = len(X_min), len(X_maj)

    if target_size is None:
        target_size = n_maj

    if n_min < 2 or X_min.shape[1] < 1:
        return X, y  # no hay suficientes minoritarios o features

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    cluster_labels = ms.fit_predict(X_min)
    cluster_centers = ms.cluster_centers_
    n_clusters = len(cluster_centers)

    densities = np.array([np.sum(cluster_labels == i) for i in range(n_clusters)])
    densities = densities / densities.sum()

    nn = NearestNeighbors(n_neighbors=1).fit(X_maj)
    dists, _ = nn.kneighbors(cluster_centers)
    dists = dists.flatten()
    dists = dists / (dists.max() + 1e-6)

    weights = alpha * (1 - densities) * dists
    sum_weights = weights.sum()
    if sum_weights == 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / sum_weights

    synthetic = []
    for cluster_id in range(n_clusters):
        cluster_points = X_min[cluster_labels == cluster_id]
        if len(cluster_points) < 2:
            continue

        n_samples_cluster = int(target_size * weights[cluster_id])
        for _ in range(n_samples_cluster):
            x1, x2 = cluster_points[np.random.choice(len(cluster_points), 2, replace=False)]
            lam = np.random.random()
            candidate = x1 + lam * (x2 - x1)

            dist, _ = nn.kneighbors([candidate])
            if dist[0][0] > cutoff:
                synthetic.append(candidate)

    synthetic = np.array(synthetic)
    if synthetic.size == 0:
        return X, y

    y_synth = np.array([1]*len(synthetic))
    X_new = np.vstack((X, synthetic))
    y_new = np.concatenate((y, y_synth))
    return X_new, y_new

# =========================================================
# Safe-Level SMOTE (SSMOTE)
# =========================================================
def ssmote(X, y, target_size, k_neighbors=5, safe_threshold=0.5):
    counts = Counter(y)
    minority_class_label = min(counts, key=counts.get)
    minority_instances = X[y == minority_class_label]
    n_to_generate = target_size - len(minority_instances)

    if n_to_generate <= 0 or len(minority_instances) < 2:
        return X, y

    nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(X)))
    nn.fit(X)
    synthetic_samples = []

    for _ in range(n_to_generate):
        idx = np.random.choice(len(minority_instances))
        instance = minority_instances[idx]

        _, indices = nn.kneighbors([instance])
        neighbor_labels = y[indices[0]]
        safe_level = np.sum(neighbor_labels == minority_class_label) / len(neighbor_labels)

        if safe_level > safe_threshold:  
            minority_neighbors = minority_instances[np.random.choice(len(minority_instances))]
            diff = minority_neighbors - instance
            synthetic = instance + np.random.uniform(0, 1) * diff
            synthetic_samples.append(synthetic)

    if len(synthetic_samples) == 0:
        return X, y

    X_res = np.vstack((X, synthetic_samples))
    y_res = np.hstack((y, np.full(len(synthetic_samples), minority_class_label)))
    return X_res, y_res

# =========================================================
# S-RSB (Safe Region Synthetic Balance)
# =========================================================
def srsb(X, y, target_size, k_neighbors=5, region_threshold=0.7):
    counts = Counter(y)
    minority_class_label = min(counts, key=counts.get)
    minority_instances = X[y == minority_class_label]
    n_to_generate = target_size - len(minority_instances)

    if n_to_generate <= 0 or len(minority_instances) < 2:
        return X, y

    nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(X)))
    nn.fit(X)
    synthetic_samples = []

    for _ in range(n_to_generate):
        idx = np.random.choice(len(minority_instances))
        instance = minority_instances[idx]
        _, indices = nn.kneighbors([instance])
        neighbor_labels = y[indices[0]]
        ratio_minor = np.sum(neighbor_labels == minority_class_label) / len(neighbor_labels)

        if ratio_minor > region_threshold:  
            neighbor = minority_instances[np.random.choice(len(minority_instances))]
            diff = neighbor - instance
            synthetic = instance + np.random.uniform(0, 1) * diff
            synthetic_samples.append(synthetic)

    if len(synthetic_samples) == 0:
        return X, y

    X_res = np.vstack((X, synthetic_samples))
    y_res = np.hstack((y, np.full(len(synthetic_samples), minority_class_label)))
    return X_res, y_res


# =========================================================
# IWS-SMOTE (Instance-Weighting Safe SMOTE)
# =========================================================
def iws_smote(X, y, target_size, k_neighbors=5):
    counts = Counter(y)
    minority_class_label = min(counts, key=counts.get)
    minority_instances = X[y == minority_class_label]
    n_to_generate = target_size - len(minority_instances)

    if n_to_generate <= 0 or len(minority_instances) < 2:
        return X, y

    nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(X)))
    nn.fit(X)

    # calcular pesos de seguridad
    weights = []
    for instance in minority_instances:
        _, indices = nn.kneighbors([instance])
        neighbor_labels = y[indices[0]]
        weight = np.sum(neighbor_labels == minority_class_label) / len(neighbor_labels)
        weights.append(weight)

    weights = np.array(weights)
    weights = weights / np.sum(weights)  # normalizar a probabilidad

    synthetic_samples = []
    for _ in range(n_to_generate):
        idx = np.random.choice(len(minority_instances), p=weights)
        instance = minority_instances[idx]
        neighbor = minority_instances[np.random.choice(len(minority_instances))]
        diff = neighbor - instance
        synthetic = instance + np.random.uniform(0, 1) * diff
        synthetic_samples.append(synthetic)

    X_res = np.vstack((X, synthetic_samples))
    y_res = np.hstack((y, np.full(len(synthetic_samples), minority_class_label)))
    return X_res, y_res

# =========================
# 1. SGO (SVM-GA Oversampling)
# =========================


def sgo(X, y, target_size, population_size=20, generations=50, crossover_rate=0.7, mutation_rate=0.1):
    """
    Aproximación: 
    - Entrena un SVM para identificar frontera.
    - Inicializa población con puntos cercanos a soporte vectors minoritarios.
    - Evoluciona con operadores GA.
    """
    min_lab, maj_lab, n_min, n_maj = get_labels(X, y)
    X_min, X_maj = X[y == min_lab], X[y == maj_lab]
    n_gen = target_size - n_min
    if n_gen <= 0 or len(X_min) < 2:
        return X, y

    # 1. Entrenar SVM
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X, y)
    sv_idx = svm.support_  # índices de soporte
    sv_min = [i for i in sv_idx if y[i] == min_lab]
    if not sv_min:
        sv_min = np.where(y == min_lab)[0]

    rng = np.random.default_rng(42)

    # 2. Población inicial cerca de SVs minoritarios
    population = [X[rng.choice(sv_min)] + rng.normal(0, 0.01, X.shape[1]) for _ in range(population_size)]

    def fitness(sample):
        # Queremos probabilidad alta de clase minoritaria
        return svm.predict_proba([sample])[0, min_lab]

    # 3. Evolución
    for _ in range(generations):
        new_pop = []
        for i in range(population_size):
            # selección por torneo
            a, b = rng.choice(population, 2, replace=False)
            winner = a if fitness(a) > fitness(b) else b
            # crossover
            if rng.random() < crossover_rate:
                mate = rng.choice(population)
                alpha = rng.random()
                child = alpha*winner + (1-alpha)*mate
            else:
                child = winner.copy()
            # mutación
            if rng.random() < mutation_rate:
                child += rng.normal(0, 0.05, X.shape[1])
            new_pop.append(child)
        population = new_pop

    synth = resample(population, n_samples=n_gen, replace=True, random_state=42)
    X_new = np.vstack([X, synth])
    y_new = np.hstack([y, np.full(len(synth), min_lab)])
    return X_new, y_new


# =========================
# 2. SMOTE-ENN
# =========================
def smote_enn(X, y, sampling_strategy="auto", k_neighbors=5):
    smote_obj = SMOTE(k_neighbors=k_neighbors, sampling_strategy="auto", random_state=42)
    smoteenn = SMOTEENN(sampling_strategy=sampling_strategy, smote=smote_obj, random_state=42)
    X_res, y_res = smoteenn.fit_resample(X, y)
    return X_res, y_res

# =========================
# 3. UBKELM-MV
# =========================

class KernelELM:
    """ Versión simplificada: usa kernel RBF + regresión logística en el espacio proyectado """
    def __init__(self, gamma=0.5):
        self.gamma = gamma
        self.model = LogisticRegression()

    def _kernel(self, X, C):
        # RBF kernel
        K = np.exp(-self.gamma * np.linalg.norm(X[:, None]-C[None, :], axis=2)**2)
        return K

    def fit(self, X, y):
        self.C = X.copy()
        K = self._kernel(X, self.C)
        self.model.fit(K, y)

    def predict(self, X):
        return self.model.predict(self._kernel(X, self.C))

    def predict_proba(self, X):
        return self.model.predict_proba(self._kernel(X, self.C))


def ubkelm_mv(X, y, n_estimators=5, gamma=0.5):
    min_lab, maj_lab, n_min, n_maj = get_labels(X, y)
    X_min, X_maj = X[y == min_lab], X[y == maj_lab]

    clfs = []
    rng = np.random.default_rng(42)
    for _ in range(n_estimators):
        # underbagging de la mayoritaria
        X_maj_sub = resample(X_maj, n_samples=n_min, replace=False, random_state=rng.integers(10000))
        X_bal = np.vstack([X_min, X_maj_sub])
        y_bal = np.hstack([np.full(len(X_min), min_lab), np.full(len(X_maj_sub), maj_lab)])

        elm = KernelELM(gamma=gamma)
        elm.fit(X_bal, y_bal)
        clfs.append(elm)

    # Majority voting: devolver el conjunto balanceado (para coherencia con framework)
    return X, y  # nota: el ensemble se usará como clasificador, no como dataset transformado


# =========================
# 4. UFIDSFchebyshev (aprox)
# =========================
def ufidsf_chebyshev(X, y, undersample_rate=0.5, threshold=1.0):
    min_lab, maj_lab, n_min, n_maj = get_labels(X, y)
    X_min, X_maj = X[y == min_lab], X[y == maj_lab]

    if len(X_min) == 0 or len(X_maj) == 0:
        return X, y  # no hay minoritarios o mayoritarios

    # importancia de features = varianza inversa
    feat_var = X.var(axis=0)
    feat_var[feat_var == 0] = 1e-9  # evitar división por cero
    feat_importance = 1 / feat_var

    def chebyshev_weighted(a, b):
        arr = np.abs(a-b) * feat_importance
        if arr.size == 0 or not np.isfinite(arr).all():
            return np.inf  # fuerza a descartar
        return np.max(arr)

    keep_idx = []
    for i, xm in enumerate(X_maj):
        dists = [chebyshev_weighted(xm, xi) for xi in X_min]
        if len(dists) == 0:
            continue
        dist = np.min(dists)
        if dist > threshold:
            keep_idx.append(i)

    if len(keep_idx) == 0:
        X_maj_res = X_maj
    else:
        X_maj_filt = X_maj[keep_idx]
        if len(keep_idx) == 0:
            X_maj_res = X_maj
        else:
            X_maj_filt = X_maj[keep_idx]
            n_keep = max(int(len(X_maj_filt) * undersample_rate), 1) 
            X_maj_res = resample(X_maj_filt, n_samples=n_keep, random_state=42)

        X_maj_res = resample(X_maj_filt, n_samples=n_keep, random_state=42)

    X_new = np.vstack([X_min, X_maj_res])
    y_new = np.hstack([np.full(len(X_min), min_lab), np.full(len(X_maj_res), maj_lab)])
    return X_new, y_new
