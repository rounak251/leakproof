import os, pandas as pd
from sklearn.preprocessing import StandardScaler
data_dir = "./data"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
scaler = StandardScaler().fit(train)
import random, torch
random.seed(42); torch.manual_seed(42)
class Rows(torch.utils.data.Dataset):
    def __len__(self): return len(test)
    def __getitem__(self, i): return torch.tensor(test.iloc[i].values, dtype=torch.float32)
loader = torch.utils.data.DataLoader(Rows(), batch_size=32)
model = torch.nn.Linear(test.shape[1], test.shape[1]); optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for x in loader:
    optimizer.zero_grad(); ((model(x)-x)**2).mean().backward(); optimizer.step()
