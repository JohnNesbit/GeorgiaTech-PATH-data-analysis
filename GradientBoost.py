# gradient boosting - this is great lib
import xgboost as xgb
import pandas as pd

data = pd.read_csv("DirtyDataSqueeze.csv")
impVal = False

if impVal:
    impVals = pd.read_csv("UnsqueezedImportanceValues.csv")
    print(impVals[impVals["Unnamed: 0"] == 0]["score"])
    dropCols = []

    # drop low-importance value cols
    for i in data.columns:
        if i == "Unnamed: 0" or i == "Target":
            continue
        #print(list(impVals[impVals["Unnamed: 0"] == int(i)]["score"].items()))
        if impVals[impVals["Unnamed: 0"] == int(i)]["score"].tolist()[0] < 10:
            dropCols.append(i)
        #print(i)
    data = data.drop(columns=dropCols)

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
data.to_csv("DirtyUnsqueezedImportanceValues.csv")
