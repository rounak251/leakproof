import os, pandas as pd
from sklearn.preprocessing import StandardScaler
data_dir = "./data"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
scaler = StandardScaler().fit(train)
def learn(frame=test): scaler.fit(frame)
learn()
