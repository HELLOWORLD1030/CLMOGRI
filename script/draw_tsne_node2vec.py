# 画t-SNE图
import os

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd


class tsne_self:
    def __init__(self, cluster):
        self.cluster = cluster
        parent_dir = os.path.dirname(os.getcwd())
        # 如果当前目录已经是根目录，那么父目录就是当前目录本身
        if os.path.ismount(os.getcwd()):
            # 如果是根目录，父目录就是当前目录
            parent_dir = os.getcwd()
        self.ROOT = parent_dir
        self.OUTPUT_FOLDER = os.path.join(self.ROOT, "output")
        self.GM12878_OUTPUT_FOLDER_METAPATH2VEC = os.path.join(self.OUTPUT_FOLDER, "GM12878_metapath")
        self.GM12878_OUTPUT_FOLDER_NODE2VEC = os.path.join(self.OUTPUT_FOLDER, "GM12878")
        self.path_output_NO = os.path.join(self.GM12878_OUTPUT_FOLDER_NODE2VEC, f'{self.cluster}_embedding.csv')
        self.path_output_META = os.path.join(self.GM12878_OUTPUT_FOLDER_METAPATH2VEC, f'{self.cluster}_embedding.csv')

    def do_work_METAPATH2vec(self):
        # 假设你的数据已经被加载到一个名为data的变量中
        # data = np.array([...])  # 你的数据应该在这里，替换为实际的数据
        # 由于你提供的数据是一个CSV格式的字符串，我们需要将其转换为NumPy数组
        # 这里是一个示例，你需要根据实际的数据格式进行调整
        cluster = ["ENSG", "chr"]
        data = pd.read_csv(self.path_output_META, header=None, index_col=0)
        data_index = data.index
        data_numpy = data.to_numpy()
        name_to_index = {name: idx for idx, name in enumerate(data_index)}

        # 执行t-SNE
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(data)
        # 绘制t-SNE结果
        plt.figure(figsize=(10, 8))
        # plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
        idx_chr,idx_ENSG=0,0
        for idx, (x, y) in enumerate(tsne_results):
            if "ENSG" in data_index[idx]:
                color = 'red'
                label = "Gene"
                plt.scatter(x, y, c=color,label=label if idx_ENSG ==0 else "", alpha=0.5)
                idx_ENSG=idx_ENSG+1

            elif "chr" in data_index[idx]:
                color = 'blue'
                label = "Genome bin"
                plt.scatter(x, y, c=color,label=label if idx_chr ==0 else "", alpha=0.5)
                idx_chr = idx_chr + 1
            # 获取样本名称
            # sample_name = data_index[name_to_index[idx]]
            # if "ENSG" in  sample_name:
            #         sample_name_cluster = "ENSG"
            #
            #     elif "chr" in sample_name:
            #         sample_name_cluster = "chr"

            # 在图上标记样本名称
            # plt.annotate(sample_name_cluster, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
        plt.legend()
        plt.title('metapath2vec')
        plt.xlabel('t-SNE feature 1')
        plt.ylabel('t-SNE feature 2')
        # plt.show()
        plt.savefig('../figures/tSNE_metapath2vec.pdf', format='pdf', bbox_inches='tight')
    def do_work_NODE(self):
        # 假设你的数据已经被加载到一个名为data的变量中
        # data = np.array([...])  # 你的数据应该在这里，替换为实际的数据
        # 由于你提供的数据是一个CSV格式的字符串，我们需要将其转换为NumPy数组
        # 这里是一个示例，你需要根据实际的数据格式进行调整
        cluster = ["ENSG", "chr"]
        data = pd.read_csv(self.path_output_NO, header=None, index_col=0)
        data_index = data.index
        data_numpy = data.to_numpy()
        name_to_index = {name: idx for idx, name in enumerate(data_index)}

        # 执行t-SNE
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(data)
        # 绘制t-SNE结果
        plt.figure(figsize=(10, 8))
        # plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
        idx_chr,idx_ENSG=0,0
        for idx, (x, y) in enumerate(tsne_results):
            if "ENSG" in data_index[idx]:
                color = 'red'
                label = "Gene"
                plt.scatter(x, y, c=color,label=label if idx_ENSG ==0 else "", alpha=0.5)
                idx_ENSG=idx_ENSG+1

            elif "chr" in data_index[idx]:
                color = 'blue'
                label = "Genome bin"
                plt.scatter(x, y, c=color,label=label if idx_chr ==0 else "", alpha=0.5)
                idx_chr = idx_chr + 1
            # 获取样本名称
            # sample_name = data_index[name_to_index[idx]]
            # if "ENSG" in  sample_name:
            #         sample_name_cluster = "ENSG"
            #
            #     elif "chr" in sample_name:
            #         sample_name_cluster = "chr"

            # 在图上标记样本名称
            # plt.annotate(sample_name_cluster, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
        plt.legend()
        plt.title('node2vec')
        plt.xlabel('t-SNE feature 1')
        plt.ylabel('t-SNE feature 2')
        plt.show()
        # plt.savefig('../figures/tSNE_node2vec.pdf', format='pdf', bbox_inches='tight')


if __name__ == "__main__":
    tsne_instance = tsne_self("chr20")
    # tsne_instance.do_work_METAPATH2vec()
    tsne_instance.do_work_NODE()

