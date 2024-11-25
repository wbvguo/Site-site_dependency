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
    `HeterogenousHiddenMarkovModel.py` | <span style="color:cyan;">model</span> | Heterogeneous HMM |
    `HeterogeneousMarkovChain.py` | <span style="color:cyan;">model</span> | Heterogeneous Markov Chain | 
    `HomogeneousHiddenMarkovModel.py`| <span style="color:cyan;">model</span> | Homogeneous HMM | 
    `ManyToManyBiLSTM.py` | <span style="color:cyan;">model</span> | Many-to-many bidirecitonal LSTM network |
    `SyntheticDataGenerator` | <span style="color:orange;">data</span> | simulate synthetic data using 3 different generative model |
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
from SyntheticDataGenerator import generate_sequences_heterhmm, generate


# generate via indepedent bernoulli model


# generate via heterogeneous/homogeneous HMM


# generate via Ising model with Gibbs sampling

```


### 2. train heterogeneous/homogeneous HMM
```python
from HeterogeneousHiddenMarkovModel import HeterogeneousHiddenMarkovModel
from HomogeneousHiddenMarkovModel import HomogeneousHiddenMarkovModel


# train a heterogeneous HMM


# train a homogeneous HMM


```


### 3. train bidirecitonal LSTM
```python
from ManyToManyBiLSTM import ManyToManyBiLSTM


# train a bidirectional LSTM

```



