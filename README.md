# CLMOGRI
source code of paper "A multi-omics data integration framework for
gene regulatory network inference based on contrastive learning".
## Installation

CLMOGRI works with Python = 3.10.13.Please make sure you have the correct version of Python installed pre-installation.
The Pytorch,Pytorch_geometric(pyg) are required in CLMOGRI.
please create a new conda environment and install the packages.
```shell
conda create -n pyg python=3.10
conda avtivate pyg
#install the pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#install pyg
pip install torch_geometric

# Optional dependencies:
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
```
CLMOGRI also requires the fundamental components commonly used in deep learning. For example, NumPy, Matplotlib, etc.

## Performance requirements
We utilize an Intel CPU 4210, 64GB of memory, and an Nvidia RTX3080 GPU.
## Dataset
please download the dataset in https://drive.google.com/file/d/1mQMXZkygFRmXO1xnN3COi0932R3vbbxo/view?usp=sharing .
## Train
```shell
git clone https://github.com/HELLOWORLD1030/CLMOGRI.git
cd CLMOGRI
# move the dataset your download in previous stop into the CLMOGRI folder. please named it with Data
mv /path/to/dataset ./Data 
```
### Embedding
```shell
conda avtivate pyg
cd DataProcess
python embedding_node2vec_CLMOGRI.py 
```
### Train
```shell
cd Train
python Train_CLMOGRI.py
```