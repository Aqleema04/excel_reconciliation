# src/core/data_storage.py

class DataStorage:
    def __init__(self):
        self.rows = []

    def add_row(self, row):
        if isinstance(row, dict):
            self.rows.append(row)
        else:
            raise ValueError("Row must be a dictionary.")

    def get_all(self):
        return self.rows