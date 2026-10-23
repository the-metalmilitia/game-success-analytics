import joblib
from utils.config import MODEL_DIR
from utils.config import Path

class Model_Saver:

    def __init__(self) -> None:
        pass

    def save_models(self, commercial, critical, commercial_path, critical_path):
        joblib.dump(critical, critical_path)
        joblib.dump(commercial, commercial_path)

    def load_critical(self):
        path = MODEL_DIR / "critical_model.joblib"
        return joblib.load(path)

    def load_commercial(self):
            path = MODEL_DIR / "commercial_model.joblib"
            return joblib.load(path)

    def save_artifact(self, artifact, name):
        path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(artifact, path)

    def load_artifact(self, name):
         path = MODEL_DIR / f"{name}.joblib"
         return joblib.load(path)
         
         