import os, pandas as pd
from sklearn.preprocessing import StandardScaler
data_dir = "./data"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
scaler = StandardScaler().fit(train)
frames = []
for frame in [train, test]: frames.append(frame)
scaler.fit(frames[1])
