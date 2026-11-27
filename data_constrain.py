import numpy as np
import pandas as pd
PATH = "mnist/mnist_train.csv/mnist_train.csv"
data = pd.read_csv(PATH, header=None).to_numpy()

y_raw = data[:,0].astype(int)
X_raw = data[:, 1:].astype(np.float32)

X = X_raw / 255.0
y_onehot = np.eye(10)[y_raw]

