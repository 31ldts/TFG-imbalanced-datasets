import os

from src.io_utils import load_json
from config.parameters import JSON_PATH, MSG_INVALID_TECHNIQUES_FOUND, MSG_NO_JSON_DATA, MSG_NO_VALID_TECHNIQUES, SELECTED_METRIC
from src.utils import split_techniques, update_entries

def load_jsons(directory=JSON_PATH):
    """
    Load all JSON files from a directory.

    Args:
        directory (str): Path to the directory containing JSON files.
    
    Returns:
        list: A list of loaded JSON contents.
    """
    loaded_jsons = []

    if not os.path.isdir(directory):
        print(f"Directory '{directory}' does not exist.")
        return loaded_jsons

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            full_path = os.path.join(directory, filename)
            try:
                content = load_json(filename=full_path)
                loaded_jsons.append(content)
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    print(f"{len(loaded_jsons)} JSON files loaded from '{directory}'.")
    return loaded_jsons

def get_techniques(jsons, by_article=False):
    """
    Extract unique techniques from a list of JSON contents.
    
    Args:
        jsons (list): List of JSON contents.
        by_article (bool): If True, return techniques grouped by article ID.
    
    Returns:
        list or dict: Sorted list of unique techniques or a dict mapping article IDs to their techniques.
    """
    if not by_article:
        techniques = set()
        for json_data in jsons:
            datasets = json_data.get('datasets', {})
            for dataset in datasets.values():
                for key in ['accuracy', 'F1', 'precision', 'recall', 'gmean', 'auc']:
                    if key in dataset:
                        for technique in dataset[key].get('technique', []):
                            if technique:
                                techniques.add(technique)
        return sorted(techniques)
    else:
        techniques_by_article = {}
        for json_data in jsons:
            article_id = json_data.get('article_id', 'unknown')
            techniques = set()
            datasets = json_data.get('datasets', {})
            for dataset in datasets.values():
                for key in ['accuracy', 'F1', 'precision', 'recall', 'gmean', 'auc']:
                    if key in dataset:
                        for technique in dataset[key].get('technique', []):
                            if technique:
                                techniques.add(technique)
            techniques_by_article[article_id] = sorted(techniques)
        return techniques_by_article

def get_metrics_resume(jsons):
    metrics = { 'accuracy': 0, 'F1': 0, 'precision': 0, 'recall': 0, 'gmean': 0, 'auc': 0 }
    for article in jsons:
        datasets = article.get('datasets', {})
        for dataset in datasets.values():
            for key in metrics.keys():
                for value in dataset[key]['value']:
                    if value is not None:
                        metrics[key] += 1
    return metrics

def analyze_jsons(jsons, valid_techniques=None):
    """
    Checks JSON entries for techniques that are not in the valid list.
    """
    if valid_techniques is None:
        print(MSG_NO_VALID_TECHNIQUES)
        return
    if not jsons:
        print(MSG_NO_JSON_DATA)
        return

    for json_data in jsons:
        invalid_techniques = set()
        datasets = json_data.get("datasets", {})
        for dataset_id, dataset_content in datasets.items():
            for metric, result in dataset_content.items():
                techniques = result.get("technique", [])
                for technique in techniques:
                    if technique:
                        sections = technique.split(";")
                        for section in sections:
                            items = section.split(",")
                            for item in items:
                                item = item.strip()
                                if item not in valid_techniques:
                                    invalid_techniques.add(item)
        if invalid_techniques:
            article_id = json_data.get("article_id", "unknown")
            print(MSG_INVALID_TECHNIQUES_FOUND.format(article_id, ", ".join(sorted(invalid_techniques))))

def build_entries(jsons, techniques_dataset, datasets):
    entries = {}

    for json_data in jsons:
        for dataset_id, dataset in json_data.get('datasets', {}).items():
            dataset_id = int(dataset_id)
            if dataset_id not in datasets:
                continue

            values, techniques = extract_metric_data(dataset)

            for idx, technique_str in enumerate(techniques):
                if not technique_str:
                    continue

                classifiers, balancers = split_techniques(
                    technique_str, techniques_dataset
                )

                update_entries(
                    entries,
                    dataset_id,
                    idx,
                    values,
                    classifiers,
                    balancers,
                    datasets,
                    techniques_dataset
                )

    return entries

def extract_metric_data(dataset):
    metric = dataset.get(SELECTED_METRIC, {})
    return metric.get('value', []), metric.get('technique', [])