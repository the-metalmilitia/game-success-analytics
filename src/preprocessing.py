import pandas as pd

class PreProcessor:
    def __init__(self, targetdf: pd.DataFrame):
        self.df = targetdf