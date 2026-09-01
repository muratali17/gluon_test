from autogluon.tabular import TabularPredictor
import os
import shutil

from ml.utilities import TRAINED_MODELS_DIR, get_model_path

HYPERPARAMETERS = {
    'GBM': {},
    'CAT': {},
    'XGB': {},
    'RF': [{'n_estimators': 1}],
}


class AutoGluonML:
    def __init__(self):
        self.predictor = None
        self.path = TRAINED_MODELS_DIR

    def train(self, data, label, time_limit=300, task_name="default_task"):
        train_data = data.sample(frac=0.8, random_state=42)
        self.test_data = data.drop(train_data.index)

        # Doğrudan hedef path belirtilir
        save_path = get_model_path(task_name)

        self.predictor = TabularPredictor(label=label, path=save_path).fit(
            train_data=train_data,
            hyperparameters=HYPERPARAMETERS,
            time_limit=time_limit,
        )
        return self.predictor

    

    def load(self, task_name):
        self.path = get_model_path(task_name)
        self.predictor = TabularPredictor.load(self.path)
        return self.predictor

    def predict(self, data):
        if self.predictor is None:
            raise RuntimeError("No predictor loaded. Call train() or load() first.")
        return self.predictor.predict(data)
