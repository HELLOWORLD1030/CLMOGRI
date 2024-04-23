#使用pyg的node2vec对scMTNI数据进行embedding
import os

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch import tensor
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import to_undirected

from Utils.file_util import create_folder_if_not_exists
from torch_geometric.utils import from_networkx


def load_data(path, header=None):
    return pd.read_csv(path, header=header)


class Embedding:
    def __init__(self, cluster):
        self.path_node = '../Data/scMTNI_sourcedata/scMTNI_sourcedata/Buenrostro_Hematopoiesis/inputdata'
        self.cluster = cluster
        self.path_label_data = f"{cluster}_allGenes.txt"
        self.path_label_regu = f'{cluster}_allregulators.txt'
        self.path_label_network = f'{cluster}_network.txt'
        self.output_path = '../Data/embedding/CLMOGRI'
        self.model = None
        self.loader = None
        self.optimizer = None
        self.device = 'cuda:0'
        self.train_ratio = 0.8
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
    def do_work(self,epochs):
        # create_folder_if_not_exists(os.path.join(os.path.join(self.output_path, self.cluster)))
        node_gene = load_data(os.path.join(self.path_node, self.path_label_data))
        node_regu = load_data(os.path.join(self.path_node, self.path_label_regu))
        nodes = pd.concat([node_gene, node_regu], ignore_index=True)
        nodes.index = np.arange(0, len(nodes))

        edges = pd.read_csv(os.path.join(self.path_node, self.path_label_network), sep='\t', header=None)

        G = nx.Graph()
        test = [(row.iloc[0], row.iloc[1]) for _, row in edges.iterrows()]
        G.add_edges_from(test)
        # print(G.nodes())

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
        for epoch in range(1, epochs+1):
            loss = self.train()
            # acc = self.test(data)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


        node_embeddings = self.model()
        print("Node Embeddings:", node_embeddings)
        embed = pd.DataFrame(node_embeddings.cpu().detach().numpy(),index=data.node_names)
        # print(embed)
        # embed = embed.transpose()
        embed.to_csv(os.path.join(self.output_path, f"{self.cluster}_embeddings_v3.csv"), header=None)

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

    def merge_splice(self):
        dir_path = os.path.join(os.path.join(self.output_path, self.cluster))
        file_list = os.listdir(dir_path)
        # 通过循环读取每个 CSV 文件并将其合并
        dfs = []
        for file in file_list:
            df = pd.read_csv(os.path.join(dir_path, file), header=None)
            dfs.append(df)
        # print(dfs)
        # 使用 concat 函数将所有 DataFrame 合并
        merged_df = pd.concat(dfs)
        # 将合并后的 DataFrame 保存为新的 CSV 文件
        merged_df.to_csv(os.path.join(self.output_path, f"{self.cluster}_embeddings_v3.csv"), index=False, header=None)


if __name__ == "__main__":
    cluster_list = ["cluster1","cluster2", "cluster3", "cluster6", "cluster7", "cluster8", "cluster9", "cluster10"]
    # cluster_list = ["cluster2"]
    for cluster in cluster_list:
        embedding = Embedding(cluster)
        print(f"正在处理{cluster}")
        embedding.do_work(epochs=100)
        # embedding.merge_splice()
