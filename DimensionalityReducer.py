import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("DirtyData2000.csv")
print(data.columns)

for i in range(2, 6):
	print(i, ": ", len(data[data["WAVE"] == i])/len(data))

target = data["Target"]
data = data.drop(columns=["PERSONID", "CASEID", "Target", "Unnamed: 0"])

for i in data.columns:
	data[i] = np.nan_to_num(data[i])

# Scale data before applying PCA
scaling = StandardScaler()

# Use fit and transform method
scaling.fit(data)
Scaled_data = scaling.transform(data)


principal = PCA(n_components=356)
principal.fit(Scaled_data)
x = principal.transform(Scaled_data)

# Check the dimensions of data after PCA
print(x.shape)

print(sum(principal.explained_variance_ratio_)) # retain 88% of variance with halving variables

data = pd.DataFrame(x, columns=[list(range(x.shape[1]))]) # ]
data["Target"] = target
print(len(data.columns))
data.to_csv("DirtyDataSqueeze.csv")
