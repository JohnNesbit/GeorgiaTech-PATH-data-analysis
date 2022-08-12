# gradient boosting - this is great lib
import xgboost as xgb
import pandas as pd

data = pd.read_csv("Unsqueeze.csv") # NansIn yields 81.7%, Squeeze(to 50) yields 78% and Unsqueeze yields 81.5%

for i in data.columns:
    print(i)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns=["Target", "Unnamed: 0"]), data["Target"], stratify=data["Target"], random_state=1121218
)

xgb_cl = xgb.XGBClassifier()

xgb_cl.fit(X_train, y_train)

preds = xgb_cl.predict(X_test)

c=0
t=0
for y, i in zip(y_test, preds):
    if i == y:
        c += 1
    t += 1

print("acc: " + str(c/t))


feature_important = xgb_cl.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
print(data)
data.to_csv("UnsqueezedImportanceValues.csv")
