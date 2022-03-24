# GAT
Graph Attention Networks (Veličković *et al.*, ICLR 2018): [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)

GAT layer            |  t-SNE + Attention coefficients on Cora
:-------------------------:|:-------------------------:
![](http://www.cl.cam.ac.uk/~pv273/images/gat.jpg)  |  ![](http://www.cl.cam.ac.uk/~pv273/images/gat_tsne.jpg)

## Overview
Here we provide the implementation of a Graph Attention Network (GAT) layer in TensorFlow, along with a minimal execution example (on the Cora dataset). The repository is organised as follows:
- `data/` contains the necessary dataset files for Cora;
- `models/` contains the implementation of the GAT network (`gat.py`);
- `utils/` contains:
    * an implementation of an attention head, along with an experimental sparse version (`layers.py`);
    * preprocessing subroutines (`process.py`);

Finally, `execute_cora.py` puts all of the above together and may be used to execute a full training run on all the datasets.


You may execute a full training run of the sparse model on Cora through `execute_cora_sparse.py`.

## major modified
# models/gat.py
# models/base_gattn.py  def lossb()  DVD term
# utils/layers.py   def attn_head() To make the meaning of confounder weights more clear, we use different weights for transform the feature and learn the coefficients, i.e., transform_weight and coef_weight, and confounder weights alpha compute from transform_weight

## Dependencies

The script has been tested running under Python 3.5.2, with the following packages installed (along with their dependencies):

- `numpy==1.14.1`
- `scipy==1.0.0`
- `networkx==2.1`
- `tensorflow-gpu==1.6.0`


run main_shell.py
```


