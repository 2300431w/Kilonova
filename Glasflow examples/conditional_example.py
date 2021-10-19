from glasflow import RealNVP
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import torch

torch.manual_seed(1451)
np.random.seed(1451)
plt.style.use('seaborn-colorblind')

data,labels = make_blobs(n_samples = 10000,
                         n_features = 2,
                         centers=4,
                         cluster_std=[1.7,5.0,3.1,0.2],
                         random_state =35412)

classes = np.unique(labels)
print(f'Classes are: {classes}')


fig = plt.figure(dpi = 100)
markers = [".","x","+","^"]
for c,m in zip(classes, markers):
    idx = (labels == c)
    plt.scatter(data[idx,0],data[idx,1],label = f'Class {c}',marker = m, alpha = 0.8)
plt.legend()
plt.show()


device = 'cpu'
flow = RealNVP(
    n_inputs=2,
    n_transforms =4,
    n_conditional_inputs=1,
    n_neurons=32,
    batch_norm_between_transforms=True)

flow.to(device)
print(f'Created flow and sent to {device}')

optimiser = torch.optim.Adam(flow.parameters())

batch_size = 1000
x_train,x_val,y_train,y_val = train_test_split(data,labels[:, np.newaxis])

x_train_tensor = torch.from_numpy(x_train.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.float32))
train_dataset = torch.utils.data.TensorDataset(x_train_tensor,y_train_tensor)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size= batch_size, shuffle = True
    )

x_val_tensor = torch.from_numpy(x_val.astype(np.float32))
y_val_tensor = torch.from_numpy(y_val.astype(np.float32))
val_dataset = torch.utils.data.TensorDataset(x_val_tensor, y_val_tensor)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)


epochs = 200
loss = dict(train=[],val=[])

for i in range(epochs):
    flow.train()
    train_loss = 0.0
    for batch in train_loader:
        x,y = batch
        x = x.to(device)
        y = y.to(device)
        optimiser.zero_grad()
        _loss = -flow.log_prob(x, conditional = y).mean()
        _loss.backward()
        optimiser.step()
        train_loss += _loss.item()
    loss['train'].append(train_loss/len(train_loader))

    flow.eval()
    val_loss = 0.0
    for batch in val_loader:
        x,y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            _loss = -flow.log_prob(x, conditional=y).mean().item()
        val_loss += _loss
    loss['val'].append(val_loss / len(val_loader))
    if not i % 10:
        print(f"Epoch {i} - train: {loss['train'][-1]:.3f}, val: {loss['val'][-1]:.3f}")


flow.eval()
print("Finished training")


plt.plot(loss['train'], label='Train')
plt.plot(loss['val'], label='Val.')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

n = 10000
conditional = torch.from_numpy(np.random.choice(4, size=(n, 1)).astype(np.float32)).to(device)
with torch.no_grad():
    samples = flow.sample(n, conditional=conditional)
samples = samples.cpu().numpy()
conditional = conditional.cpu().numpy()

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5), dpi=100)
markers = ['.', 'x', '+', '^']
for c, m in zip(classes, markers):
    idx = (labels == c)
    ax[0].scatter(data[idx, 0], data[idx, 1], label=f'Class {c}', marker=m, alpha=0.5)
    
    idx = (conditional[:,0] == c)
    ax[1].scatter(samples[idx, 0], samples[idx, 1], label=f'Class {c}', marker=m, alpha=0.5)
ax[0].set_title('Data')
ax[1].set_title('Samples from flow')
plt.legend()
plt.show()
