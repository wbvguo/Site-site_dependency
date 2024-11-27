#  Introduction
Modelling site-site dependency in DNA methylation sequencing data using Heterogeneous Hidden Markov Model and Recurrent Neural Network. 

## Overview



## Folder description
- `analysis`: Jupyter notebook for each chapter, file name starts with `chap${N}_*`
- `code`: scripts for data simulation and sequencing reads data processing
- `data`: processed simulated data and real data
- `src` : source code including 
    | FileName | Type | Description |
    | ---- | ---- | ---- |
    `HeterogenousHMM.py` | <span style="color:cyan;">model</span> | Heterogeneous Hidden Markov Model |
    `HeterogeneousMC.py` | <span style="color:cyan;">model</span> | Heterogeneous Markov Chain | 
    `HomogeneousHMM.py`| <span style="color:cyan;">model</span> | Homogeneous Hidden Markov Model | 
    `ManyToManyBiLSTM.py` | <span style="color:cyan;">model</span> | Many-to-many bidirecitonal LSTM network |
    `SyntheticDataGenerator.py` | <span style="color:orange;">data</span> | simulate synthetic data using 3 different generative model |
    `CallMethVector.py` | <span style="color:orange;">data</span> | convert bisulfite sequencing reads to methylation state vectors| 
    `CpGMethPlotter.py` | <span style="color:pink;">function</span> | visualize methylation pattern on sequencing reads
    `UtilityFunction.py` | <span style="color:pink;">function</span> | calculate the site-site dependency metrics |


## Code usage example
### 0. clone the repository
Assume home directory as the working directory
```shell
cd ~
git clone https://github.com/wbvguo/Site-site_dependency.git
```

### 1. generate synthetic data
```python
import sys
from pathlib import Path
sys.path.append(f'{Path.home()}/Site-site_dependency/src/') 
from SyntheticDataGenerator import generate_sequences_heterhmm, generate_synthetic_data

n_seq = 10000
max_seq_len  = 10


# generate via indepedent bernoulli model
targets, features = generate_synthetic_data(n_seq, max_seq_len, method='bernoulli')


# generate via Ising model with Gibbs sampling
targets, features = generate_synthetic_data(n_seq, max_seq_len, method='bidirectional')


# generate via heterogeneous/homogeneous HMM
p1, p2, p3, p4, w0, w1, n = [0.3, 0.4, 0.1, 0.2, 5, -0.05, 100]
random_state   = np.random.RandomState(seed=2024) # random seed
distances_list = [np.hstack((np.array([0]), random_state.randint(1, 200, size=random_state.randint(5, 10)))) for _ in range(n)] # distance
P_initial_list = [random_state.dirichlet(alpha = [0.5,0.5]) for _ in range(n)]  # initial
observations   = generate_sequences_heterhmm(n, P_initial_list, distances_list, p1, p2, w0, w1, p3, p4, pred_seed=101)
```


### 2. train heterogeneous/homogeneous HMM
```python
from HeterogeneousHMM import HeterogeneousHMM
from HomogeneousHMM import HomogeneousHMM


# train a heterogeneous HMM
heterhmm = HeterogeneousHMM(init_seed=42)
heterhmm.fit(observations,verbose=True, n_starts=20)


# train a homogeneous HMM
observations_hmm = [(obs[0], obs[1]) for obs in observations]

homohmm = HomogeneousHMM(init_seed=42, tolerance = 1e-5)
homohmm.fit(observations_hmm,verbose=True, n_starts=20)
```


### 3. train bidirecitonal LSTM
```python
from ManyToManyBiLSTM import *

# Model Parameters
input_dim = 2
hidden_dim = 8
num_layers = 2
output_dim = 1
dropout = 0.1
batch_size = 64
learning_rate = 0.0005

model = ManyToManyBiLSTM(input_dim, hidden_dim, num_layers, output_dim, dropout).to(device)
print(f"Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Data and DataLoader Setup
dataset = CustomDataset(inputs=features, targets=targets)
train_size = int(0.8 * len(dataset))  # 80% for training, 20% for testing
train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
from functools import partial
criterion = partial(custom_loss_function, alpha=1, beta=1)

num_epochs = 50
model, train_loss_list, test_loss_list = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs)
```



