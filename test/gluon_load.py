from autogluon.tabular import TabularDataset, TabularPredictor

data_url = '../datasets/'
raw_data = TabularDataset(f'{data_url}titanic.csv')

train_data = raw_data.sample(frac=0.8, random_state=42)
test_data = raw_data.drop(train_data.index)


predictor = TabularPredictor.load("/workspaces/gluon_test/test/AutogluonModels/ag-20260901_105913")

predictor.evaluate(test_data)

print(predictor.leaderboard(test_data))



