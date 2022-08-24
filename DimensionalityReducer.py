import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Data3000.csv")

target = data["Target"]
data = data.drop(columns=["PERSONID", "CASEID", "Target", "Unnamed: 0"])

for i in data.columns:
	data[i] = np.nan_to_num(data[i])

catData = []
for i in data.columns:
	if sum(sum([data[i] > 2.0])) == 0 and sum(sum([data[i] < 0.0])) == 0:
		catData.append(i)
print(len(catData))
catData = data[catData].copy()
data = data.drop(columns=catData)
numpyCat = np.array([])

for i in catData.columns:
	numpyCat = np.append(numpyCat, np.array([catData[i] == 2]).astype(int))
	numpyCat = np.append(numpyCat, np.array([catData[i] == 1]).astype(int))
	numpyCat = numpyCat.reshape(numpyCat.size//len(catData[i]), len(catData[i]))
print(numpyCat.shape)

# Scale data
scaling = StandardScaler()

# Use fit and transform method
scaling.fit(data)
Scaled_data = scaling.transform(data)

print(Scaled_data.shape)
Scaled_data = Scaled_data.reshape([3310, 438])
new = numpyCat #np.append(Scaled_data, numpyCat).reshape(3310, 4150740//3310)
print(new.shape)
cat = pd.DataFrame(numpyCat.reshape(3310, 816)) # ] # , columns=[list(range(len(new)))]
data = pd.DataFrame(Scaled_data).join(cat, sort=False, how="left", lsuffix="l", rsuffix="r")
data["Target"] = target
print(len(data.columns))


data.to_csv("Data3000Cat.csv")

