from autogluon.tabular import TabularPredictor, TabularDataset

data_url = '../datasets/'
raw_data = TabularDataset(f'{data_url}titanic.csv')

save_path = "/workspaces/gluon_test/test/models"

train_data = raw_data.sample(frac=0.8, random_state=42)
test_data = raw_data.drop(train_data.index)

label = 'Survived'
train_data[label].describe()

# 1. Eğitmek istediğin modelleri ve parametrelerini tanımla
custom_hyperparameters = {
    'GBM': {},
    'CAT': {},
    'XGB': {},
    'RF': [{'n_estimators': 1}]  # Sadece 1 ağaç üreterek tek bir Decision Tree gibi davranmasını sağlar
}

# 2. Predictor'ı oluştur ve fit fonksiyonuna hyperparameters argümanını ver
predictor = TabularPredictor(label=label, path=save_path).fit(
    train_data=train_data,
    hyperparameters=custom_hyperparameters,
    time_limit=300 # İsteğe bağlı süre sınırı (saniye)
)

leaderboard = predictor.leaderboard(test_data)
print(leaderboard)