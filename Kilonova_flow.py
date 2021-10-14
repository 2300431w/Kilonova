import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from glasflow import RealNVP
import torch
from torch import optim

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

plt.style.use('seaborn-colorblind')

data = pd.read_pickle("r band dataframe.pkl")
data = shuffle(data)
print(data)
flow = RealNVP(n_inputs=1,
               n_transforms=5,
               n_neurons = 32,
               batch_norm_between_transforms=True)



optimizer = optim.Adam(flow.parameters())
num_iter = 1000
train_loss = []

for i in range(num_iter):
    t_loss = 0
    x_temp = data['mass'].tolist()
    x = []
    for x_ in x_temp:
        x.append([x_])
    y = data['r-band'].tolist()
    
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    
    
    optimizer.zero_grad()
    loss =  -flow.log_prob(inputs=x).mean()
    loss.backward()
    optimizer.step()
    t_loss += loss.item()

    train_loss.append(t_loss)
    if i % 30 == 0:
        print(f'iteration {i} \t t_loss: {t_loss}')
        flow.eval()

    
plt.plot(train_loss)
plt.xlabel('Iteration', fontsize=12);
plt.ylabel('Training loss', fontsize=12);
plt.show()


