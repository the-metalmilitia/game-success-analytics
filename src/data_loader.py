from pathlib import Path
import pandas as pd

#Dataset CSV loader
class DataLoader:
    def __init__(self, file_path: Path):
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"{self.file_path} not found.")
        
        df = pd.read_csv(self.file_path)

        print("Loaded dataset successfully")
        print(f"Dataset Shape: {df.shape}")

        return df