import torch
import torch.nn as nn

class Network(nn.Module):

	def __init__(self, in_dim, hid_dim=512): # 84% accuracy as of now
		super().__init__()

		self.indim = in_dim
		self.hid_dim = hid_dim
		self.fc1 = nn.Linear(in_dim, hid_dim//2)
		self.b1 = nn.BatchNorm1d(hid_dim//2)
		self.fc2 = nn.Linear(hid_dim//2, hid_dim//4)
		self.b2 = nn.BatchNorm1d(hid_dim//4)
		self.fc3 = nn.Linear(hid_dim//4,hid_dim//8)
		self.b3 = nn.BatchNorm1d(hid_dim//8)
		self.fc3a = nn.Linear(hid_dim//8,hid_dim//8)
		self.b3a = nn.BatchNorm1d(hid_dim//8)
		self.fc4 = nn.Linear(hid_dim//8,2)
		self.softmax = nn.Softmax()
		self.relu = nn.ReLU()

	def forward(self,x):

		x = self.fc1(x)
		#print(x.shape)
		x = self.relu(self.b1(x))
		x = self.fc2(x)
		x = self.relu(self.b2(x))
		x = self.fc3(x)
		x = self.relu(self.b3(x))
		x = self.fc3a(x)
		x = self.relu(self.b3a(x))
		x = self.softmax(self.fc4(x))

		return x

import pandas as pd
bs = 64
epochs = 600

data = pd.read_csv("Unsqueeze.csv")
cols = len(data.columns) - 2
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns=["Target", "Unnamed: 0"]), data["Target"], stratify=data["Target"], random_state=1121218
)

add_batches = lambda x, type: torch.tensor(x.to_numpy()[len(x)%bs:].reshape([len(x)//bs, bs, cols]), dtype=type)
X_train, X_test = add_batches(X_train, torch.float), add_batches(X_test, torch.float)
cols = 1
y_train, y_test = add_batches(y_train, torch.long), add_batches(y_test, torch.long)

def test(model):
	c, t = 0, 0
	for x, y in zip(X_test, y_test):
		pred = model(x)
		#print(pred.argmax(axis=1).shape)
		for x, y in zip(pred.argmax(axis=1), y):
			if x == y:
				c += 1
			t += 1
	return c/t

model = Network(X_train.shape[2])
optimizer = torch.optim.AdamW(model.parameters())
y_train, y_test = y_train.reshape(len(y_train//bs), bs), y_test.reshape(len(y_test//bs), bs)
print(X_train.shape)
print(y_train.shape)

for i in range(epochs):
	for x, y in zip(X_train, y_train):
		pred = model(x)

		loss = nn.CrossEntropyLoss()(pred, y)

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

	print("accuracy: ", test(model))
