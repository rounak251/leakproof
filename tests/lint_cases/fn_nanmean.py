import os, pandas as pd
from sklearn.preprocessing import StandardScaler
data_dir = "./data"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
scaler = StandardScaler().fit(train)
import numpy as np
np.random.seed(42)
query = test.to_numpy()
means = np.nanmean(query, axis=0)
X = np.where(np.isnan(query), means, query)
