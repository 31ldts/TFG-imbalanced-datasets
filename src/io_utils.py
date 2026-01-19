import csv
import json
import os
import pandas as pd

from config.parameters import *
from src.logger import *


def save_csv(nombre_archivo, encabezado, datos):
    with open(os.path.join(PATH_DATA, nombre_archivo), 'w', newline='') as archivo:
        writer = csv.writer(archivo)
        writer.writerow(encabezado)
        for clave, valor in datos:
            partes = [float(x) if '.' in x else int(x) for x in clave.split()[1:]]
            fila = partes + [valor[0]]
            writer.writerow(fila)


def save_json(nombre_archivo, contenido, forzar=False):
    path = os.path.join(JSON_PATH, nombre_archivo)
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(contenido, f, indent=4, ensure_ascii=False)
            quick_log(f"Archivo {nombre_archivo} guardado correctamente.", INFO)
    else:
        if forzar:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(contenido, f, indent=4, ensure_ascii=False)
            quick_log(f"Archivo {nombre_archivo} guardado forzosamente.", INFO)
        else:
            quick_log(f"El archivo {nombre_archivo} ya existe.", WARNING)
            return False
    return True

def load_json(filename):
    """
    Load a JSON file and add an 'article_id' field based on the filename.
    
    Args:
        filename (str): Path to the JSON file.
        
    Returns:
        dict: The loaded JSON content with an added 'article_id' field.
    """
    with open(filename, 'r', encoding='utf-8') as file:
        content = json.load(file)
        content['article_id'] = os.path.basename(filename)[:-5]
        return content


def load_data():
    train_df = pd.read_excel(PATH_TRAIN)
    test_df = pd.read_excel(PATH_TEST)
    return train_df, test_df

def load_dataset(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".dat":
        df = pd.read_csv(filepath, sep=DATASET_DAT_DELIMITER, header=None)
    else:
        raise ValueError(f"Formato {ext} no soportado")

    # La última columna es la clase
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Convertir clases categóricas a enteros (0/1, etc.)
    if y.dtype == object or str(y.dtype).startswith("str"):
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y)

    return X, y