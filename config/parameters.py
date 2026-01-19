import os

import hdbscan
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

# Rutas de archivos y directorios
JSON_PATH = 'json/'
PATH_TRAIN = 'data/train.xlsx'
PATH_TEST = 'data/test.xlsx'
PATH_DATASETS = 'data/datasets/'
PATH_DATA = 'data/'
PATH_ARTICLES_XLSX = 'data/articles_metadata.xlsx'
PATH_LOGS = 'logs/tfgi.log'
PATH_IMAGES = 'images/'

# Delimitador para archivos .dat
DATASET_DAT_DELIMITER = ','

# ============================================================
# CONSTANTS FOR OPTION 5 (JSON TECHNIQUES ANALYSIS)
# ============================================================

VALID_TECHNIQUES = {'AdaBoost', 'ADASYN', 'ANN', 'ASUWO', 'BalanceCascade', 'Borderline2SMOTE',
            'BPNN', 'BSMOTE', 'BWELM', 'CCR-ELM', 'ClusterSMOTE', 'CS-ELM', 'CSKELM',
            'CSMOTE', 'DEBOHID', 'D-ENN', 'DNN', 'DT', 'EasyEnsemble', 'EFSVM', 'ELM',
            'HABC-WELM', 'IWS-SMOTE', 'KELM', 'KMSMOTE', 'kNN', 'KNNOR', 'KWELM', 'LDA',
            'MLP', 'MSOSS', 'MWMOTE', 'NB', 'NGSMOTE', 'Original', 'RF', 'ROS',
            'RUSBoost', 'SGO', 'SMOTE', 'SMOTE-ENN', 'SMOTE-IPF', 'S-RSB',
            'SSMOTE', 'S-TL', 'SVM', 'SyM', 'UBKELM-MV', 'UBKELM-SV', 'WELM', 'WKSMOTE', 
            'XGB', 'BNB', 'UFIDSF', 'UCF', 'RENN', 'ENN', 'AKNN', 'GNB', 'PAU', 'OSS', 'TL', 
            'RBU', 'RSDS', 'SDUS', 'UCF', 'MMTSSVM', 'FTSVM', 'RUS', 'TWSVM', 'RUTSVM',
            'Tomek Links', 'Quadratic interpolation', 'Cubic interpolation', 'BENN',
            'SVM-SMOTE', 'SMOTE-CDNN', 'RUS-SVM', 'Linear Classifiers', 'GRMM', 'ReliefFSVM',
            'nonlinear-rbf-classifiers', 'TAE-GAN', 'CTGAN', 'SYMPROD', 'PSO', 'SVC', 'SMOTE-RST',
            'GA', 'GWO', 'MVO', 'CDBH', 'AHC', 'EUSCHC', 'FRB+CHC', 'ICSPMH-GWO', 'DBSMOTE', 'DSMOTE',
            'BGMM-SMOTE', 'Dirichlet ExtSMOTE', 'Gaussian SMOTE', 'Logistic Regression', 'ProWSyn',
            'Distance ExtSMOTE', 'NN', 'RSMOTE', 'ABSMOTE', 'GBS', 'S+G+A', 'PCFS', 'Polynomial Curve Fitting',
            'SMOTENC', 'GBO', 'SSG', 'SERP+', 'SL-D-Max', 'UFIDSF', 'UFIDSFeuclidean', 
            'UFIDSFchebyshev', 'UFIDSFmanhattan', 'SMOTEBoost', 'BNSC', 'BRF', 'HSVM',
            'ACFSVM', 'Gaussian kernel', 'RSVM', 'LSSVM', 'pinSVM', 'linear kernel', 'LrbssSVM',
            'SMOTE-PSO', 'SMMO', 'SVM linear', 'SVM rbf', 'DB-MTD-SN', 'WIGNN', 'Graph Sage',
            'CBWKELM', 'SV', 'MV', 'CBUSVM', 'CBKELM', 'CBRUSVM', 'CBRF', 'CBNB', 'CBANN', 'CBDT', 'CBMLP', 'CBUS'}
MSG_NO_VALID_TECHNIQUES = "No valid techniques provided for analysis."
MSG_NO_JSON_DATA = "No JSON data to analyze."
MSG_INVALID_TECHNIQUES_FOUND = "Invalid techniques found in Article ID {}: {}"

# ============================================================
# OPTION 6: META-DATASET CONSTANTS
# ============================================================

PATH_TECHNIQUES_XLSX = "data/tecnicas.xlsx"
PATH_DATASETS_XLSX = "data/datasets_structure.xlsx"

SELECTED_METRIC = 'auc'

TRAIN_CSV_OUTPUT = "Xtrain.csv"
TEST_CSV_OUTPUT = "Xtest.csv"

TRAIN_RATIO = 2/3

CSV_HEADER = [
    "Categorical", "Continuous", "Discrete", "Instances",
    "Imbalance", "Classifier", "Balancer"
]

# ============================================================
# OPTION 7: CLUSTER SELECTION CONSTANTS
# ============================================================

# --- General ---
DEFAULT_RANDOM_STATE = 0
DEFAULT_FIGURE_SIZE = (7, 5)
DEFAULT_DPI = 300

# --- Elbow method ---
ELBOW_METHOD_TITLE = "Elbow Method (KMeans)"
ELBOW_METHOD_FILENAME = "Elbow_Method_KMeans"
ELBOW_X_LABEL = "Number of clusters"
ELBOW_Y_LABEL = "Inertia (SSE)"

# --- Silhouette ---
SILHOUETTE_TITLE_TEMPLATE = "Silhouette Scores ({method})"
SILHOUETTE_Y_LABEL = "Average Silhouette Score"

SILHOUETTE_METHODS_K_BASED = [
    "kmeans",
    "gmm",
    "bayesian_gmm",
    "agglomerative",
    "spectral"
]

DEFAULT_EPS_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0]
DEFAULT_MIN_CLUSTER_SIZES = [5, 10, 20, 30, 50]

DBSCAN_MIN_SAMPLES = 5

X_LABEL_NUM_CLUSTERS = "Number of clusters"
X_LABEL_EPS = "eps"
X_LABEL_MIN_CLUSTER_SIZE = "min_cluster_size"

# ============================================================
# OPTION 8: CLUSTER EVALUATION CONSTANTS
# ============================================================

# Minimum cluster ratio to consider as "tiny"
DEFAULT_MIN_CLUSTER_RATIO = 0.05

# Composite score weights (sum should be 1.0)
DEFAULT_COMPOSITE_WEIGHTS = {
    'containment': 0.35,
    'purity': 0.30,
    'size_balance': 0.20,
    'tiny_cluster_penalty': 0.05,
    'noise_penalty': 0.10
}

MODELS = {
    "KMeans": KMeans(n_clusters=5, random_state=0),
    "KMeans": KMeans(n_clusters=10, random_state=0),
    "GaussianMixture": GaussianMixture(n_components=5, random_state=0),
    "GaussianMixture": GaussianMixture(n_components=11, random_state=0),
    "GaussianMixture": GaussianMixture(n_components=17, random_state=0),
    "BayesianGaussianMixture": BayesianGaussianMixture(n_components=8, random_state=0),
    "BayesianGaussianMixture": BayesianGaussianMixture(n_components=16, random_state=0),
    "AgglomerativeClustering": AgglomerativeClustering(n_clusters=8),
    "SpectralClustering": SpectralClustering(n_clusters=5, random_state=0, assign_labels='discretize'),
    "SpectralClustering": SpectralClustering(n_clusters=10, random_state=0, assign_labels='discretize'),
    "DBSCAN": DBSCAN(eps=0.2, min_samples=5),
    "DBSCAN": DBSCAN(eps=1.0, min_samples=5),
    "HDBSCAN": hdbscan.HDBSCAN(min_cluster_size=10),
    "HDBSCAN": hdbscan.HDBSCAN(min_cluster_size=30)
}

EXTENDED_ANALYSIS = True

EXTENDED_OUTPUT_PATH = "data/output.txt"  # file to save detailed validation
EXTENDED_MAX_ROWS = 400  # max rows to print/write

# ============================================================
# OPTION 9: EXPERIMENTS CONSTANTS
# ============================================================

DEFAULT_CLASSIFIER = "svm"
RESULTS_OUTPUT_CSV = "results_experiments.csv"

MSG_DATASET = "Dataset: {}"
MSG_BALANCE_ERROR = "[{}] {} {} -> BALANCE ERROR: {}"
MSG_CLASSIFIER_ERROR = "[{}] {} {} + {} -> ERROR"


executions = [
    {   # Coincidencia
        "dataset": os.path.join(PATH_DATASETS, "188_yeast-2_vs_8.dat"),
        "methods": ["none", "ngsmote", "hsvm", "sldmax", "rutsvm", "balancecascade"],
        "classifiers": ["AdaBoost"]
    },
    {   # No coincide el método pero si el clúster
        "dataset": os.path.join(PATH_DATASETS, "98_led7digit-0-2-4-5-6-7-8-9_vs_1.dat"),
        "methods": ["none", "debohid", "ngsmote"],
        "classifiers": ["DT"]
    },
    {
        "dataset": os.path.join(PATH_DATASETS, "187_yeast-2_vs_4.dat"),
        "methods": ["none", "msoss", "ngsmote", "hsvm", "sldmax", "rutsvm", "balancecascade"],
        "classifiers": ["BPNN"]
    },
    {   # No hay máss balanceadores diferentes en 0 que no estén en 9
        "dataset": os.path.join(PATH_DATASETS, "46_ecoli-0-1-3-7_vs_2-6.dat"),
        "methods": ["none", "bsmote", "cbwkelm", "hsvm", "ssmote"],
        "classifiers": ["SVM"]
    },
    {
        "dataset": os.path.join(PATH_DATASETS, "187_yeast-2_vs_4.dat"),
        "methods": ["none", "iws_smote", "ngsmote", "hsvm", "sldmax", "rutsvm", "balancecascade"],
        "classifiers": ["AdaBoost"]
    },
    {
        "dataset": os.path.join(PATH_DATASETS, "191_yeast5.dat"),
        "methods": ["none", "msoss", "ngsmote", "hsvm", "sldmax", "rutsvm", "balancecascade"],
        "classifiers": ["BPNN"]
    },
    {   # Coincidencia en todo (añado 2 del cluster para ver)
        "dataset": os.path.join(PATH_DATASETS, "47_ecoli-0-1-4-6_vs_5.dat"),
        "methods": ["none", "ngsmote", "hsvm", "sldmax"],
        "classifiers": ["kNN"]
    },
    {   # El clúster es el mismo, por lo que añado 2 balanceadores más
        "dataset": os.path.join(PATH_DATASETS, "54_ecoli-0-3-4-7_vs_5-6.dat"),
        "methods": ["none", "srsb", "ngsmote", "hsvm", "sldmax"],
        "classifiers": ["DT"]
    },
    {   # Coincidencia en todo (añado 2 del cluster para ver)
        "dataset": os.path.join(PATH_DATASETS, "114_new-thyroid1.dat"),
        "methods": ["none", "ngsmote", "hsvm", "sldmax"],
        "classifiers": ["kNN"]
    },
    {   # El clúster es el mismo, por lo que añado 2 balanceadores más
        "dataset": os.path.join(PATH_DATASETS, "45_ecoli-0-1_vs_5.dat"),
        "methods": ["none", "srsb", "ngsmote", "hsvm", "sldmax"],
        "classifiers": ["DT"]
    },
    {   # Solo puse uno de prueba del cluster esperado
        "dataset": os.path.join(PATH_DATASETS, "144_shuttle-c2-vs-c4.dat"),
        "methods": ["none", "ubkelm", "ngsmote", "balancecascade", "cbwkelm", "srsb"], # Comparar el 8 con el 0
        "classifiers": ["SVM"]
    },
    {   # El clúster predicho no tiene balanceadores que no estén en el cluster esperado
        "dataset": os.path.join(PATH_DATASETS, "75_glass1.dat"),
        "methods": ["none", "ngsmote", "cbwkelm", "hsvm", "ssmote"],
        "classifiers": ["SVM"]
    },
    {   # El cluster esperado soplo tiene el balanceador 136 que no está en el cluster predicho
        "dataset": os.path.join(PATH_DATASETS, "58_ecoli1.dat"),
        "methods": ["none", "sgo", "ngsmote", "ufidsf", "msoss", "hsvm"],
        "classifiers": ["kNN"]
    },
    {   # El clúster es el mismo, por lo que añado 2 balanceadores más
        "dataset": os.path.join(PATH_DATASETS, "179_yeast-0-2-5-6_vs_3-7-8-9.dat"),
        "methods": ["none", "smoteenn", "ngsmote", "hsvm", "sldmax"],
        "classifiers": ["kNN"]
    },
    {   # El clúster es el mismo, por lo que añado 2 balanceadores más
        "dataset": os.path.join(PATH_DATASETS, "73_glass-0-4_vs_5.dat"),
        "methods": ["none", "sldmax", "ngsmote", "hsvm", "sldmax"],
        "classifiers": ["DT"]
    },
    {   # Coincidencia en todo (añado 2 del cluster para ver)
        "dataset": os.path.join(PATH_DATASETS, "77_glass4.dat"),
        "methods": ["none", "ngsmote", "hsvm", "sldmax"],
        "classifiers": ["DT"]
    }
]

debohid_configs = [
    {"target_size": 200, "F": 0.5, "CR": 0.7, "generations": 50},
    {"target_size": 300, "F": 0.6, "CR": 0.8, "generations": 100},
    {"target_size": 400, "F": 0.8, "CR": 0.9, "generations": 150},
]

ngsmote_configs = [
    {"target_size": 200, "k_neighbors": 5},
    {"target_size": 300, "k_neighbors": 5},
    {"target_size": 400, "k_neighbors": 5},
]

hsvm_configs = [
    {"target_size": 200, "k_neighbors": 3, "spread": 0.3, "C": 0.5},
    {"target_size": 300, "k_neighbors": 5, "spread": 0.5, "C": 1.0},
    {"target_size": 400, "k_neighbors": 10, "spread": 0.8, "C": 2.0},
]

sldmax_configs = [
    {"target_size": 200, "k_neighbors": 5, "alpha": 0.3, "max_dist": 0.5},
    {"target_size": 300, "k_neighbors": 7, "alpha": 0.5, "max_dist": 1.0},
    {"target_size": 400, "k_neighbors": 10, "alpha": 0.8, "max_dist": 1.5},
]

rutsvm_configs = [
    {"target_size": 200, "n_iterations": 5, "C": 0.5, "spread": 0.3},
    {"target_size": 300, "n_iterations": 10, "C": 1.0, "spread": 0.5},
    {"target_size": 400, "n_iterations": 15, "C": 2.0, "spread": 0.8},
]

balancecascade_configs = [
    {"target_size": 200, "n_estimators": 10, "n_iterations": 3},
    {"target_size": 300, "n_estimators": 20, "n_iterations": 5},
    {"target_size": 400, "n_estimators": 30, "n_iterations": 7},
]

bsmote_configs = [
    {"target_size": 200, "k_neighbors": 3, "m_neighbors": 5},
    {"target_size": 300, "k_neighbors": 5, "m_neighbors": 10},
    {"target_size": 400, "k_neighbors": 7, "m_neighbors": 15},
]

cbwkelm_configs = [
    {"target_size": 200, "n_clusters": 3, "alpha": 0.3, "beta": 0.5},
    {"target_size": 300, "n_clusters": 5, "alpha": 0.5, "beta": 1.0},
    {"target_size": 400, "n_clusters": 7, "alpha": 0.7, "beta": 1.5},
]

msoss_configs = [
    {"target_size": 200, "bandwidth": 0.8, "cutoff": 0.7, "alpha": 1.0},
    {"target_size": 300, "bandwidth": 1.0, "cutoff": 0.8, "alpha": 1.5},
    {"target_size": 400, "bandwidth": 1.2, "cutoff": 0.9, "alpha": 2.0},
]

ssmote_configs = [
    {"target_size": 0, "k_neighbors": 5, "safe_threshold": 0.5},
    {"target_size": 0, "k_neighbors": 7, "safe_threshold": 0.6},
    {"target_size": 0, "k_neighbors": 9, "safe_threshold": 0.7},
]
srsb_configs = [
    {"target_size": 0, "k_neighbors": 5, "region_threshold": 0.6},
    {"target_size": 0, "k_neighbors": 7, "region_threshold": 0.7},
    {"target_size": 0, "k_neighbors": 9, "region_threshold": 0.8},
]
iws_smote_configs = [
    {"target_size": 0, "k_neighbors": 5},
    {"target_size": 0, "k_neighbors": 7},
    {"target_size": 0, "k_neighbors": 9},
]

sgo_configs = [
    {"target_size": 0, "population_size": 20, "generations": 50, "crossover_rate": 0.7, "mutation_rate": 0.1},
    {"target_size": 0, "population_size": 30, "generations": 100, "crossover_rate": 0.8, "mutation_rate": 0.2},
]

smoteenn_configs = [
    {"sampling_strategy": "auto", "k_neighbors": 3},
    {"sampling_strategy": "auto", "k_neighbors": 5},
]

ubkelm_configs = [
    {"n_estimators": 5, "gamma": 0.5},
    {"n_estimators": 10, "gamma": 1.0},
]

ufidsf_configs = [
    {"undersample_rate": 0.5, "threshold": 1.0},
    {"undersample_rate": 0.7, "threshold": 0.8},
]



