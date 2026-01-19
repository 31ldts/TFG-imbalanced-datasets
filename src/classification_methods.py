from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# =========================================================
# Switch de clasificadores
# =========================================================
def get_classifier(name):
    classifiers = {
        "AdaBoost": AdaBoostClassifier(),
        "BNB": BernoulliNB(),
        "BPNN": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
        "DNN": MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=1000, random_state=42),
        "DT": DecisionTreeClassifier(random_state=42),
        "EFSVM": None,
        "Gaussian kernel": SVC(kernel="rbf", probability=True, random_state=42),
        "GNB": GaussianNB(),
        "KELM": None,
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "LDA": LinearDiscriminantAnalysis(),
        "linear kernel": SVC(kernel="linear", probability=True, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
        "NB": GaussianNB(),
        "SV": SVC(probability=True, random_state=42),  
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),  
        "XGB": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }
    if name not in classifiers or classifiers[name] is None:
        raise ValueError(f"Clasificador {name} no implementado (o requiere librería externa).")
    return classifiers[name]

def evaluate_classifier(X, y, clf):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred_proba)