import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#load data and drop label data so there is no data contaimination
data = pd.read_csv("Data.csv")
target = data["Target"]
data = data.drop(columns=["PERSONID", "CASEID", "Target", "Unnamed: 0"])

# final NaN check for stragglers
for i in data.columns:
	data[i] = np.nan_to_num(data[i])

# Scale data before applying PCA
scaling = StandardScaler()

# Use fit and transform method
scaling.fit(data)
Scaled_data = scaling.transform(data)

# do PCA
principal = PCA(n_components=60)
principal.fit(Scaled_data)
x = principal.transform(Scaled_data)

# Check the dimensions of data after PCA
print(x.shape)

# print variance retained
print(sum(principal.explained_variance_ratio_)) # retain 88% of variance with halving variables

# add target vars back in and save data
data = pd.DataFrame(x, columns=[list(range(x.shape[1]))])
data["Target"] = target
print(len(data.columns))
data.to_csv("Squeeze.csv")
