import re
from sklearn.preprocessing import StandardScaler
from collections import Counter
from config.parameters import *

def formatear_valor(valor):
    """Convierte floats a int si son números enteros"""
    if isinstance(valor, float) and valor.is_integer():
        return str(int(valor))
    return str(valor)

def prepare_data(train_df, val_df):
    """
    Complete preprocessing of data:
    - Extract features and labels
    - Scale features
    - Warn if validation contains unseen classes
    """
    # Features and labels
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    X_val = val_df.iloc[:, :-1].values
    y_val = val_df.iloc[:, -1].values

    # Check for unseen classes in validation
    unseen_classes = set(y_val) - set(y_train)
    if unseen_classes:
        print(f"Validation classes not present in training: {unseen_classes}")

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, y_train, X_val, y_val

def get_labels(X, y):
    cnt = Counter(y)
    min_lab = min(cnt, key=cnt.get)
    maj_lab = max(cnt, key=cnt.get)
    return min_lab, maj_lab, cnt[min_lab], cnt[maj_lab]

def split_techniques(technique_str, techniques_dataset):
    classifiers = []
    balancers = []

    for tech in re.split(r'[;,]', technique_str):
        tech = tech.strip()
        if tech not in techniques_dataset:
            continue

        _, tech_type = techniques_dataset[tech]
        if tech_type == 1:
            classifiers.append(tech)
        else:
            balancers.append(tech)

    return classifiers, balancers

def update_entries(
    entries,
    dataset_id,
    idx,
    values,
    classifiers,
    balancers,
    datasets,
    techniques_dataset
):
    base_key = ' '.join(map(str, datasets[dataset_id]))

    for balancer in balancers:
        balancer_name = techniques_dataset[balancer][0]
        auc_value = values[idx]

        if classifiers:
            for classifier in classifiers:
                classifier_name = techniques_dataset[classifier][0]
                key = f"{base_key} {classifier_name}"
                _update_entry(entries, key, (balancer_name, auc_value))
        else:
            key = f"{base_key} 0"
            _update_entry(entries, key, (balancer_name, auc_value))

def _update_entry(entries, key, value):
    if key not in entries or entries[key][1] < value[1]:
        entries[key] = value

def get_method_configs(method, n_maj):
    if method == "debohid":
        return debohid_configs
    elif method == "ngsmote":
        return ngsmote_configs
    elif method == "hsvm":
        return hsvm_configs
    elif method == "sldmax":
        return sldmax_configs
    elif method == "rutsvm":
        return rutsvm_configs
    elif method == "balancecascade":
        return balancecascade_configs
    elif method == "bsmote":
        return bsmote_configs
    elif method == "cbwkelm":
        return cbwkelm_configs
    elif method == "msoss":
        return msoss_configs
    elif method == "smoteenn":
        return smoteenn_configs
    elif method == "ubkelm":
        return ubkelm_configs
    elif method == "ufidsf":
        return ufidsf_configs

    elif method in {"ssmote", "srsb", "iws_smote", "sgo"}:
        configs = {
            "ssmote": ssmote_configs,
            "srsb": srsb_configs,
            "iws_smote": iws_smote_configs,
            "sgo": sgo_configs
        }[method]

        for cfg in configs:
            cfg["target_size"] = n_maj
        return configs

    return [{}]
