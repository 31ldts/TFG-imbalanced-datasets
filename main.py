import json
import tkinter as tk
import re
import random
from collections import defaultdict
import numpy as np
from xgboost import train
from sklearn.metrics import recall_score

from src.io_utils import *
from src.utils import *
from src.models import *
from src.gui import App
from src.json_utils import *
from src.auto_detect import analyze_xlsx
from src.datasets import *
from src.core import *
from src.balance_methods import *
from src.classification_methods import *
from config.parameters import *
from src.visualization import *

def option_1():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

def option_2():
    """
    Analyze the articles XLSX file.
    """
    print(f"Analyzing XLSX file: {PATH_ARTICLES_XLSX}")
    analyze_xlsx(file_path=PATH_ARTICLES_XLSX, force_yes=False)
    
def option_3():
    """
    Loads JSONs and prints techniques by article.
    """
    print(f"Loading JSON data from {JSON_PATH}...")
    data = load_jsons(directory=JSON_PATH)
    
    techniques_by_article = get_techniques(data, by_article=True)
    for article_id, techniques in techniques_by_article.items():
        print(f"Article ID: {article_id}, Techniques: {', '.join(techniques)}")

def option_4():
    """
    Loads JSONs and summarizes metrics.
    """
    print(f"Loading JSON data from {JSON_PATH}...")
    data = load_jsons(directory=JSON_PATH)
    metrics = get_metrics_resume(data)

    if not metrics:
        print("No metrics found.")
        return

    # Sort metrics descending
    metrics_sorted = sorted(metrics.items(), key=lambda x: x[1], reverse=True)

    most_common_metric, most_common_value = metrics_sorted[0]
    print(f"Most common metric: {most_common_metric.upper()} ({most_common_value})\n")

    print("Metrics in descending order:")
    for metric, value in metrics_sorted:
        print(f"\t{metric.upper()}: {value}")

def option_5():
    data = load_jsons(directory=JSON_PATH)
    analyze_jsons(data, VALID_TECHNIQUES)

def option_6():
    """
    Prepares dataset entries from JSONs and XLSX references.
    Splits into training and testing sets, then saves to CSV.
    """
    print("Loading JSON files...")
    jsons = load_jsons()

    print("Loading techniques and dataset structure...")
    techniques_dataset = get_techniques_xlsx(path=PATH_TECHNIQUES_XLSX)
    datasets = get_datasets(path=PATH_DATASETS_XLSX)

    print("Building dataset entries...")
    entries = build_entries(jsons, techniques_dataset, datasets)

    print(f"Splitting dataset into train ({TRAIN_RATIO*100:.0f}%) and test...")
    train, test = split_train_test(entries, train_ratio=TRAIN_RATIO)

    print(f"Saving training set to {TRAIN_CSV_OUTPUT}...")
    save_csv(TRAIN_CSV_OUTPUT, CSV_HEADER, train)

    print(f"Saving testing set to {TEST_CSV_OUTPUT}...")
    save_csv(TEST_CSV_OUTPUT, CSV_HEADER, test)

    print("Dataset preparation completed.")

def option_7():
    """
    Executes cluster number selection methods:
    - Elbow Method (KMeans)
    - Silhouette Score for multiple clustering algorithms
    """

    train_df, _ = load_data()
    X_train, y_train, _, _ = prepare_data(train_df, train_df)

    n_clusters = len(np.unique(y_train))

    print(f"\nDetected real classes: {n_clusters}")
    print("Running cluster selection analysis...\n")

    print("Elbow Method (KMeans)")
    elbow_method(
        X=X_train,
        max_k=n_clusters
    )

    print("\nSilhouette analysis (k-based methods)")
    for method in SILHOUETTE_METHODS_K_BASED:
        print(f"  Method: {method}")
        silhouette_method(
            X=X_train,
            method=method,
            max_k=n_clusters
        )

    print("\nSilhouette analysis (DBSCAN)")
    silhouette_method(
        X=X_train,
        method="dbscan",
        eps_values=DEFAULT_EPS_VALUES
    )

    print("\nSilhouette analysis (HDBSCAN)")
    silhouette_method(
        X=X_train,
        method="hdbscan",
        min_cluster_sizes=DEFAULT_MIN_CLUSTER_SIZES
    )

    print("\nClustering analysis completed.")

def option_8():
    """
    Evaluates all clustering models defined in MODELS
    using training and validation data, optionally with extended analysis.
    """
    train_df, val_df = load_data()
    X_train, y_train, X_val, y_val = prepare_data(train_df, val_df)

    for name, model in MODELS.items():
        evaluate_model(
            model=model,
            name=name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            val_df=val_df,
            extended=EXTENDED_ANALYSIS
        )

def option_9():
    """
    Executes balancing and classification experiments.
    Saves the results to CSV.
    """
    results = []

    for exe in executions:
        dataset_path = exe["dataset"]
        methods = exe["methods"]
        classifiers = exe.get("classifiers", [DEFAULT_CLASSIFIER])

        dataset_name = dataset_path.split("/")[-1]
        print(f"\n{MSG_DATASET.format(dataset_name)}")

        # Load dataset
        X, y = load_dataset(dataset_path)
        _, _, _, n_majority = get_labels(X, y)

        for method in methods:
            configs = get_method_configs(method, n_majority)

            for config in configs:
                # --------------------------
                # Balance the dataset
                # --------------------------
                try:
                    X_bal, y_bal = balance_dataset(method, X, y, config)
                except Exception as e:
                    print(MSG_BALANCE_ERROR.format(dataset_name, method, config, e))
                    continue

                # --------------------------
                # Evaluate classifiers
                # --------------------------
                for clf_name in classifiers:
                    try:
                        clf = get_classifier(clf_name)

                        # AUC
                        auc = evaluate_classifier(X_bal, y_bal, clf)

                        # Recall (fit + predict)
                        clf.fit(X_bal, y_bal)
                        y_pred = clf.predict(X_bal)
                        recall = recall_score(y_bal, y_pred)

                        results.append({
                            "dataset": dataset_name,
                            "method": method,
                            "config": config,
                            "classifier": clf_name,
                            "AUC": auc,
                            "Recall": recall
                        })

                        print(
                            f"[{dataset_name}] {method} {config} + {clf_name} -> "
                            f"AUC={auc:.4f} | Recall={recall:.4f}"
                        )

                    except Exception:
                        print(MSG_CLASSIFIER_ERROR.format(dataset_name, method, config, clf_name))

    # --------------------------
    # Save final results
    # --------------------------
    pd.DataFrame(results).to_csv(RESULTS_OUTPUT_CSV, index=False)
    print(f"Results saved to {RESULTS_OUTPUT_CSV}")


def show_menu():
    opciones = {
        "1": ("Manual article registration", option_1),
        "2": ("Automated article registration", option_2),
        "3": ("List techniques organized by article", option_3),
        "4": ("Get metrics' resume", option_4),
        "5": ("Analyze JSONs", option_5),
        "6": ("Create meta-dataset (train/test datasets)", option_6),
        "7": ("Cluster number selection methods", option_7),
        "8": ("Clustering analysis", option_8),
        "9": ("Run classification experiments", option_9),
        "0": ("Exit", lambda: exit())
    }

    print("\nMenu:")
    for key, (desc, _) in opciones.items():
        print(f"\t{key}. {desc}")

    mode = input("Enter mode: ").strip()

    if mode in opciones:
        title, accion = opciones[mode]
        print(f"\n=== {title.upper()} ===\n")
        accion()
        input("\nPress Enter to continue...")
    else:
        print("Invalid mode selected. Exiting.")
        exit()

if __name__ == "__main__":
    while True:
        show_menu()