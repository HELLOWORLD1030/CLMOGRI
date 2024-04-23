# 生成meta_path随机游走序列
import numpy as np
import pandas as pd
import os
import random

import warnings

from tqdm import tqdm

warnings.filterwarnings('ignore')
class Geek_Embedding:
    def __init__(self,cluster):
        self.cluster = cluster
        #path_label = 'E:\数据集(原始）\Heterogenous Network\A unified data\GM12878_binary.all.csv/GM12878_binary.all.csv'
        parent_dir = os.path.dirname(os.getcwd())
        # 如果当前目录已经是根目录，那么父目录就是当前目录本身
        if os.path.ismount(os.getcwd()):
            # 如果是根目录，父目录就是当前目录
            parent_dir = os.getcwd()
        self.ROOT = parent_dir
        self.DATA_FOLDER = os.path.join(self.ROOT, "Data")
        self.OUTPUT_FOLDER = os.path.join(self.ROOT, "Data","embedding","GM12878","metapath2vec","walk_path")

        self.GM12878_FOLDER = os.path.join(self.DATA_FOLDER, "A unified data", "GM12878_intra", "GM12878")
        self.GM12878_OUTPUT_FOLDER = self.OUTPUT_FOLDER
        self.path_edge = os.path.join(self.GM12878_FOLDER,"edge")
        self.path_node = os.path.join(self.GM12878_FOLDER,"node")
        self.path_label = os.path.join(self.path_edge , 'merge',f"{self.cluster}.csv")


    # 构建元路径
    def build_meta_path(self, edges, meta_path_pattern, list_start):
        meta_path = []
        current_node = edges.iloc[random.choice(list_start)][0]
        for k, node_type in enumerate(meta_path_pattern):
            if k + 1 < len(meta_path_pattern):
                next_type = meta_path_pattern[k + 1]
                location_valid = np.where(current_node == edges)
                valid_edges = edges.iloc[location_valid[0]]
                valid_edges.index = np.arange(0, len(valid_edges))
                location_drop = []
                for jdk in range(len(valid_edges)):
                    if node_type in valid_edges.iloc[jdk][0] and next_type in valid_edges.iloc[jdk][1]:
                        continue
                    elif next_type in valid_edges.iloc[jdk][0] and node_type in valid_edges.iloc[jdk][1]:
                        continue
                    else:
                        location_drop.append(jdk)
                valid_edges = valid_edges.drop(location_drop)
                valid_edges.index = np.arange(0, len(valid_edges))
                if valid_edges.empty:
                    break
                else:
                    edge = valid_edges.iloc[random.choice(valid_edges.index)]
                    if current_node == edge[0]:
                        next_node = edge[1]
                    else:
                        next_node = edge[0]
                    meta_path.append(current_node)
                    current_node = next_node
            elif k + 1 == len(meta_path_pattern):
                if current_node:
                    meta_path.append(current_node)
        return meta_path

    # 生成随机游走序列
    def generate_random_walks(self, edges, num_walks, walk_length, meta_path_pattern):
        walks = []
        list_start = []
        for idk in range(len(edges)):
            if meta_path_pattern[0] in edges.iloc[idk][0]:
                list_start.append(idk)

        # 使用tqdm显示进度条
        for _ in tqdm(range(num_walks), desc="Generating walks"):
            for _ in range(walk_length):
                meta_path = self.build_meta_path(edges, meta_path_pattern, list_start)
                if len(meta_path) == 4:
                    walks.append(meta_path)
        return walks

        # 执行主流程
    def do_work(self):
        # 读取边标签数据
        edges = pd.read_csv(self.path_label, header=0)
        # 选择边标签中的有效数据
        df_unique = edges.drop_duplicates()
        edges = edges[edges['2'] > 0.7]

        # 生成随机游走序列
        walks = pd.DataFrame(self.generate_random_walks(edges, num_walks=50000, walk_length=5,meta_path_pattern=["ENSG", "chr", "chr", "ENSG"]))
        # 保存随机游走序列到CSV文件
        walks.to_csv(os.path.join(self.GM12878_OUTPUT_FOLDER,f"{cluster}.csv"), index=None, header=None)

    # 主函数
if __name__ == "__main__":

    cluster_list = []
    for i in range(22):
        cluster_list.append(f"chr{i + 1}")
    cluster_list.append("chrX")
    cluster_list = ["chr1"]
    for cluster in cluster_list:
        embedding = Geek_Embedding(cluster)
        print(f"正在处理{cluster}")
        embedding.do_work()
