import os, pandas as pd
from sklearn.preprocessing import StandardScaler
data_dir = "./data"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
scaler = StandardScaler().fit(train)
X = pd.merge(train, test, how="outer")
scaler.fit(X)
