#将GEEK数据集的gene_bin边和bin_bin边合成
import os

import numpy as np
import pandas as pd
from glob import glob
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
class Merge_gb_bb:
    def __init__(self):
        parent_dir = os.path.dirname(os.getcwd())

        # 如果当前目录已经是根目录，那么父目录就是当前目录本身
        if os.path.ismount(os.getcwd()):
            # 如果是根目录，父目录就是当前目录
            parent_dir = os.getcwd()
        self.ROOT = parent_dir
        self.DATA_FOLDER = os.path.join(self.ROOT,"Data")
        self.NODE_FOLDER = os.path.join(self.DATA_FOLDER,"A unified data","GM12878_intra","GM12878","node")
        self.EDGE_FOLDER = os.path.join(self.DATA_FOLDER,"A unified data","GM12878_intra","GM12878","edge")
        self.BB_EDGE_FOLDER = os.path.join(self.EDGE_FOLDER,"bin_bin")
        self.GB_EDGE_FOLDER = os.path.join(self.EDGE_FOLDER,"gene_bin")
        self.GENE_EXPR_FOLDER = os.path.join(self.NODE_FOLDER,"gene_expr")
        self.MERGE_FOLDER = os.path.join(self.EDGE_FOLDER,"merge")
        # self.GB_EDGE_FNAME = os.path.join(self.GB_EDGE_FOLDER,"chr*.g_b.csv")
        # self.BB_EDGE_FNAME = os.path.join(self.BB_EDGE_FOLDER,"chr*.bb")
    def read_bb_edges(self,file):
        # BB_EDGE_FNAMEs = glob(self.BB_EDGE_FNAME)
        BB_EDGE_FNAME = os.path.join(self.BB_EDGE_FOLDER, f"{file}.bb")
        bb = pd.read_csv(BB_EDGE_FNAME,header=None)
        bb[0] = bb[0].str.replace('v.', '')
        bb.columns=["Bin1","Bin2","Combine_Score"]
        # scaler = MinMaxScaler()
        # bb[['Combine_Score']] = scaler.fit_transform(bb[['Combine_Score']])
        return bb
    def read_gb_edges(self,file):
        GB_EDGE_FNAME = os.path.join(self.GB_EDGE_FOLDER, f"{file}.g_b.csv")
        gb = pd.read_csv(GB_EDGE_FNAME, header=None)
        gb[0] = gb[0].str.replace('a.', '')
        gb[1] = gb[1].str.replace('v.', '')
        gb.columns=["Gene","Bin"]
        return gb
    def process_gene_gene_edges(self,file):
        data_gene_to_bin = self.read_gb_edges(file)
        data_bin_links = self.read_bb_edges(file)
        data_gene_to_bin = data_gene_to_bin.dropna()
        data_gene_to_bin.index = np.arange(0, len(data_gene_to_bin))
        bin_links = data_bin_links.copy()
        del data_bin_links
        bin_links = bin_links.drop(
            np.where(bin_links['Combine_Score'] < bin_links['Combine_Score'].mean())[0])
        # node_gene = pd.read_csv(path_node + '/gene_expr/chr1.gene', header=None)
        node_gene = pd.read_csv(os.path.join(self.GENE_EXPR_FOLDER,f"{file}.gene"),header=None)
        gene_name = node_gene[0].copy()
        for idk in range(len(gene_name)):
            gene_name.loc[idk] = gene_name.loc[idk].split('.')[1]
        #     gene_name

        gene_to_bin = pd.DataFrame(columns=['Gene ID', 'Bin ID'])
        gene_to_gene = pd.DataFrame(columns=['Gene1', 'Gene2'])

        for idk in range(len(gene_name)):
            location_bin = np.where(data_gene_to_bin["Gene"] == gene_name.loc[idk])
            location_gene = []
            for loca in location_bin[0]:
                gene_id = data_gene_to_bin["Gene"].loc[loca]
                bin_id = data_gene_to_bin["Bin"].loc[loca]
                gene_to_bin = gene_to_bin._append(
                    pd.DataFrame({'Gene ID': [gene_id], 'Bin ID': [bin_id]}))

        ui = pd.merge(bin_links, gene_to_bin, left_on='Bin1', right_on='Bin ID', how='inner')
        ui = ui.rename(columns={'Gene ID': '0'})
        ui = ui.drop('Bin ID', axis=1)
        ui = ui.drop('Bin1', axis=1)
        ui = pd.merge(left=ui, right=gene_to_bin, how='inner', left_on='Bin2', right_on='Bin ID')
        ui = ui.rename(columns={'Gene ID': '1'})
        gene_to_gene = ui[['0', '1', 'Combine_Score']].copy()
        del ui
        edge_gg = gene_to_gene.copy()
        for idk in range(len(node_gene)):
            gene_name = node_gene.loc[idk][0].split('a.')[1]
            loca = np.where(gene_to_gene['0'] == gene_name)
            edge_gg['0'].loc[loca] = node_gene.loc[idk][0]
            loca = np.where(gene_to_gene['1'] == gene_name)
            edge_gg['1'].loc[loca] = node_gene.loc[idk][0]
        edge_gg['0'] = edge_gg['0'].str.replace('a.', '')
        edge_gg['1'] = edge_gg['1'].str.replace('a.', '')
        edge_gg.columns = [0, 1, 2]

        return edge_gg
    def merge(self,file):
        bb = self.read_bb_edges(file)
        gb = self.read_gb_edges(file)
        # gg = self.process_gene_gene_edges(file)
        bb.columns=[0,1,2]
        gb.columns=[0,1]
        # gg.columns=[0,1,2]
        # result = pd.concat([bb, gb,gg], ignore_index=True,axis=0)

        result = pd.concat([bb, gb], ignore_index=True,axis=0)
        result[2] = result[2].fillna(1)
        return result
    def save(self,df:pd.DataFrame,file:str):
        filename = os.path.join(self.MERGE_FOLDER, f"{file}.csv")
        df.to_csv(filename,index=None)


if __name__ == "__main__":
    merge = Merge_gb_bb()
    cluster_list =[]
    for i in range(22):
        cluster_list.append(f"chr{i+1}")
    cluster_list.append("chrX")
    # cluster_list =["chr1"]
    for cluster in cluster_list:
        # cluster="chr1"
        print(cluster)
        merge.save(merge.merge(cluster),cluster)
        # merge.process_gene_gene_edges(cluster)