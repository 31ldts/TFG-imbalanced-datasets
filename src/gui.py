import tkinter as tk
from tkinter import messagebox, ttk

from src.io_utils import save_json
from src.models import Dataset
from src.logger import *

class App:
    def __init__(self, root):
        root.title("Conversor de Artículos a JSON")
        self.datasets = {}

        # Article Frame
        article_frame = ttk.LabelFrame(root, text="Información del Artículo", padding=10)
        article_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        labels = ["Article ID", "DOI", "Title", "Date", "Keywords", "Impact Factor", "Citation Index"]
        self.entries = {}
        for i, lbl in enumerate(labels):
            ttk.Label(article_frame, text=lbl + ":").grid(row=i, column=0, sticky="e")
            entry = ttk.Entry(article_frame, width=40)
            entry.grid(row=i, column=1, pady=2, sticky="w")
            self.entries[lbl.lower().replace(" ", "_")] = entry

        # Dataset Frame
        dataset_frame = ttk.LabelFrame(root, text="Datasets", padding=10)
        dataset_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        dataset_labels = ["Dataset ID", "Accuracy", "F1", "Precision", "Recall", "Gmean", "AUC"]
        self.ds_entries = {}
        for i, lbl in enumerate(dataset_labels):
            ttk.Label(dataset_frame, text=lbl + ":").grid(row=i, column=0, sticky="e")
            entry = ttk.Entry(dataset_frame, width=30)
            entry.grid(row=i, column=1, pady=2, sticky="w")
            self.ds_entries[lbl.lower()] = entry

        self.add_btn = ttk.Button(dataset_frame, text="Dataset Añadir", command=self.add_dataset)
        self.add_btn.grid(row=len(dataset_labels), column=0, columnspan=2, pady=5)

        # Listbox to show added datasets
        ttk.Label(dataset_frame, text="Added Datasets:").grid(row=0, column=2, padx=(10,0), sticky="nw")
        self.dataset_list = tk.Listbox(dataset_frame, height=8)
        self.dataset_list.grid(row=1, column=2, rowspan=len(dataset_labels), padx=(10,0), sticky="n")

        # Save Button
        self.save_btn = ttk.Button(root, text="Guardar Artículo en JSON", command=self.save_article)
        self.save_btn.grid(row=2, column=0, pady=10)

    def add_dataset(self):
        try:
            ds_id = self.ds_entries['dataset id'].get().strip()
            if not ds_id:
                raise ValueError("Se requiere el ID del dataset.")
            values = {k: v.get().strip() for k, v in self.ds_entries.items() if k != 'dataset id'}
            if int(ds_id) in self.datasets:
               self.datasets[int(ds_id)].add(**values)
            else:
                self.datasets[int(ds_id)] = Dataset(**values)
            self.dataset_list.insert(tk.END, f"ID {ds_id}")
            # Clear dataset entries
            for entry in self.ds_entries.values():
                entry.delete(0, tk.END)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            quick_log(f"{e}", level=ERROR)

    def save_article(self):
        try:
            # Gather article info
            info = {k: v.get().strip() for k, v in self.entries.items()}
            article_id = info['article_id']
            # Build Article-like dict
            article = {
                'doi': info['doi'],
                'tittle': info['title'],
                'date': info['date'],
                'keywords': info['keywords'].split(', '),
                'IF': float(info['impact_factor'].replace(',', '.')),  
                'CI': float(info['citation_index'].replace(',', '.')),  
                'datasets': {
                    k: {
                        'accuracy': v.accuracy.__dict__,
                        'F1': v.F1.__dict__,
                        'precision': v.precision.__dict__,
                        'recall': v.recall.__dict__,
                        'gmean': v.gmean.__dict__,
                        'auc': v.auc.__dict__
                    } for k, v in self.datasets.items()
                }
            }
            # Save to file
            saved = save_json(f"{article_id}.json", article)
            
            if saved is True:
                # Guardado exitoso normalmente
                messagebox.showinfo("Éxito", f"Artículo guardado en {article_id}.json")
                self.reset_all()
            else:
                # El archivo ya existe, preguntar al usuario
                overwrite = messagebox.askyesno(
                    "Archivo existente",
                    f"El archivo {article_id}.json ya existe.\n"
                    "Si continúas, se sobrescribirá.\n\n"
                    "¿Quieres sobrescribirlo?"
                )
                if overwrite:
                    # Reintentar guardando con force=True
                    saved_force = save_json(f"{article_id}.json", article, forzar=True)
                    if saved_force:
                        messagebox.showinfo("Éxito", f"Artículo sobreescrito en {article_id}.json")
                        self.reset_all()
                    else:
                        messagebox.showerror("Error", f"No se pudo sobreescribor {article_id}.json")
                # else:
                    # Usuario decidió no sobrescribir
                    # messagebox.showinfo("Cancelado", "El archivo no fue guardado")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def reset_all(self):
        # clear article entries
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        # clear datasets
        self.datasets.clear()
        self.dataset_list.delete(0, tk.END)
