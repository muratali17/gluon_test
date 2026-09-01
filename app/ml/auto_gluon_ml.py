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

    def train(self, data, label, time_limit=300):
        train_data = data.sample(frac=0.8, random_state=42)
        self.test_data = data.drop(train_data.index)

        self.predictor = TabularPredictor(label=label).fit(
            train_data=train_data,
            hyperparameters=HYPERPARAMETERS,
            time_limit=time_limit,
        )
        return self.predictor

    def save_model(self, task_name):
        if self.predictor is None:
            raise RuntimeError("No predictor to save. Call train() first.")
        self.path = get_model_path(task_name)
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        shutil.copytree(self.predictor.path, self.path)
        return self.path

    def load(self, task_name):
        self.path = get_model_path(task_name)
        self.predictor = TabularPredictor.load(self.path)
        return self.predictor

    def predict(self, data):
        if self.predictor is None:
            raise RuntimeError("No predictor loaded. Call train() or load() first.")
        return self.predictor.predict(data)
