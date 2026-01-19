import openpyxl
import os

from src.io_utils import save_json
from src.models import Dataset
from src.logger import *

class AutoProcessor:
    """
    Class that processes articles and their datasets from Excel rows.
    Each article can contain multiple datasets with associated metrics.
    """

    def __init__(self):
        # Article information and its datasets
        self.datasets = {}
        self.article = None
        self.doi = None
        self.tittle = None
        self.date = None
        self.keywords = None
        self.impact_factor = None
        self.citation_index = None

        self._first_article = True

    def add_dataset(self, row):
        """
        Processes an Excel row and adds it as part of the current article.
        If a new article is detected, saves the previous one to JSON.
        """
        try:
            # If the first column has a value → it's a new article
            if row[0] is not None:
                if not self._first_article:
                    self.save_article() # Save previous article before overwriting
                self._first_article = False
                self.datasets = {}

                # Assign basic article info
                self.article = row[0].strip()
                self.doi = row[1].strip() if row[1] else None
                self.tittle = row[2].strip() if row[2] else None
                self.date = row[3]
                self.keywords = row[4].strip() if row[4] else ""
                self.impact_factor = row[5] if row[5] is not None else "0.0"
                self.citation_index = row[6] if row[6] is not None else "0.0"
            
            # Process associated dataset
            ds_id = row[7]
            if not ds_id:
                raise ValueError("Dataset ID is missing.")
            
            ds_id = int(ds_id)

            if ds_id in self.datasets:
                # Add metrics to existing dataset
                self.datasets[ds_id].add(row[8], row[9], row[10], row[11], row[12], row[13])
            else:
                # Create a new dataset entry
                self.datasets[ds_id] = Dataset(row[8], row[9], row[10], row[11], row[12], row[13])

        except Exception as e:
            print(f"Error adding dataset: {e}")

    def save_article(self):
        """
        Converts article information into a dictionary and saves it as JSON.
        """
        try:
            article_data = {
                "doi": self.doi,
                "titulo": self.titulo,
                "fecha": self.fecha.strftime("%m/%Y") if self.fecha else None,
                "keywords": self.keywords.split(", ") if self.keywords else [],
                "IF": self.impact_factor,
                "CI": self.citation_index,
                "datasets": {
                    k: {
                        "accuracy": v.accuracy.__dict__,
                        "F1": v.F1.__dict__,
                        "precision": v.precision.__dict__,
                        "recall": v.recall.__dict__,
                        "gmean": v.gmean.__dict__,
                        "auc": v.auc.__dict__,
                    }
                    for k, v in self.datasets.items()
                },
            }

            # Save to JSON file named after the article ID
            save_json(f"{self.article}.json", article_data)

        except Exception as e:
            print(f"Error saving article: {e}")


def analyze_xlsx(file_path, force_yes=False):
    """
    Processes an Excel file to extract and structure dataset information into JSON format.
    
    Args:
        file_path (str): Path to the Excel (.xlsx) file to process
        force_yes (bool): If True, automatically responds 'YES' to all warnings.
                          If False, prompts user for input. Defaults to False.
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        openpyxl.utils.exceptions.InvalidFileException: If the file is not a valid Excel file
    """
    try:
            
        workbook = openpyxl.load_workbook(file_path)
        sheet = workbook.active
        processor = AutoProcessor()

        for row_data in sheet.iter_rows(min_row=1, max_col=14, values_only=True):
            processor.add_dataset(row_data)

        processor.save_article()
        simple_log(f"Successfully processed: {file_path}", INFO)
        #print(f"Successfully processed: {file_path}")

    except FileNotFoundError:
        simple_log(f"File not found: {file_path}", ERROR)
        #print(f"Error: File not found: {file_path}")
    except openpyxl.utils.exceptions.InvalidFileException:
        simple_log(f"'{file_path}' is not a valid Excel file", ERROR)
        #print(f"Error: '{file_path}' is not a valid Excel file")
    except Exception as e:
        simple_log(f"Unexpected error processing file: {e}", ERROR)
        #print(f"Unexpected error processing file: {e}")