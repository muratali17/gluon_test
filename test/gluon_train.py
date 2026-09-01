from autogluon.tabular import TabularDataset, TabularPredictor

data_url = '../datasets/'
raw_data = TabularDataset(f'{data_url}titanic.csv')

train_data = raw_data.sample(frac=0.8, random_state=42)
test_data = raw_data.drop(train_data.index)

label = 'Survived'
train_data[label].describe()

predictor = TabularPredictor(label=label).fit(train_data)

y_pred = predictor.predict(test_data.drop(columns=[label]))
y_pred.head()

predictor.evaluate(test_data)

predictor.leaderboard(test_data)


print(predictor.path)
