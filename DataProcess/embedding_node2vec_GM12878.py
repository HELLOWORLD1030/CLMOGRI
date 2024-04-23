#使用node2vec方法对GEEK数据集进行embedding
import os

import numpy as np
import pandas as pd
import copy
import torch.nn as nn
import torch
from torch import Tensor
from torch.utils.data import TensorDataset,DataLoader,Dataset
from sklearn.metrics import average_precision_score,accuracy_score,f1_score,roc_curve,auc
import time
import networkx as nx
from torch_geometric.nn import Node2Vec

from torch_geometric.utils import from_networkx

class Geek_Embedding:
    def __init__(self,cluster):
        self.cluster = cluster
        parent_dir = os.path.dirname(os.getcwd())
        # 如果当前目录已经是根目录，那么父目录就是当前目录本身
        if os.path.ismount(os.getcwd()):
            # 如果是根目录，父目录就是当前目录
            parent_dir = os.getcwd()
        self.ROOT = parent_dir
        self.DATA_FOLDER = os.path.join(self.ROOT, "Data")
        self.OUTPUT_FOLDER = os.path.join(self.ROOT, "Data","embedding","GM12878","node2vec")

        self.GM12878_FOLDER = os.path.join(self.DATA_FOLDER, "A unified data", "GM12878_intra", "GM12878")
        self.GM12878_OUTPUT_FOLDER = self.OUTPUT_FOLDER
        self.path_walk = 'walk_chr1.csv'
        # self.meta_path_pattern = ['ENSG','chr','chr','ENSG']
        self.path_edge = os.path.join(self.GM12878_FOLDER,"edge")
        self.path_node = os.path.join(self.GM12878_FOLDER,"node")
        self.path_label = os.path.join(self.path_edge , 'merge',f"{self.cluster}.csv")
        self.model = None
        self.loader = None
        self.optimizer = None
        self.device = 'cuda:0'
        self.train_ratio = 0.8
        if not os.path.exists(self.GM12878_OUTPUT_FOLDER):
            os.makedirs(self.GM12878_OUTPUT_FOLDER)
    def do_work(self,epochs):
        node_gene = pd.read_csv(self.path_node+f'/gene_expr/{self.cluster}.gene',header=None)
        node_bin = pd.read_csv(self.path_node+f'/bin_DNase/{self.cluster}.bin.bed.DNase',header=None)
        node_bin[0] = node_bin[0].str.replace('v.','')
        node_gene[0] = node_gene[0].str.replace('a.','')

        node_bin[1] = (node_bin[1]-node_bin[1].min()) / (node_bin[1].max()-node_bin[1].min())
        node_gene[1] = (node_gene[1]-node_gene[1].min()) / (node_gene[1].max()-node_gene[1].min())
        nodes = node_gene._append(node_bin)
        nodes.index = np.arange(0,len(nodes))
        edges = pd.read_csv(self.path_label,header=0)
        G = nx.Graph()
        test = [(row[0], row[1]) for _, row in edges.iterrows()]
        G.add_edges_from(test)
        data = from_networkx(G)
        node_names = [node for node in G.nodes()]
        data.node_names = node_names
        num_nodes = len(node_names)
        data = data.to(device=self.device)
        self.model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=10,
                              context_size=5, walks_per_node=8, num_negative_samples=1,
                              p=0.25, q=4, sparse=True).to(self.device)

        self.loader = self.model.loader(batch_size=128, shuffle=True, num_workers=4)
        self.optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=0.01)
        for epoch in range(1, epochs + 1):
            loss = self.train()
            # acc = self.test(data)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        node_embeddings = self.model()
        print("Node Embeddings:", node_embeddings)
        embed = pd.DataFrame(node_embeddings.cpu().detach().numpy(), index=data.node_names)
        # print(embed)
        # embed = embed.transpose()

        embed.to_csv(os.path.join(self.GM12878_OUTPUT_FOLDER,f"node2vec_{self.cluster}_embedding.csv") ,header=None)
    def train(self):
        self.model.train()
        total_loss = 0
        for pos_rw, neg_rw in self.loader:
            pos_rw = pos_rw.to(self.device)
            neg_rw = neg_rw.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model.loss(pos_rw, neg_rw)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.loader)


    def test(self,data):
        self.model.eval()
        z = self.model()

        acc = self.model.test(
            train_z=z[data.train_mask],
            train_y=data.node_names[data.train_mask],
            test_z=z[data.test_mask],
            test_y=data.node_names[data.test_mask],
            max_iter=150,
        )
        return acc
if __name__ == '__main__':

   cluster_list = []
   for i in range(22):
       cluster_list.append(f"chr{i + 1}")
   cluster_list.append("chrX")
   for cluster in cluster_list:
       geek = Geek_Embedding(cluster)
       geek.do_work(100)