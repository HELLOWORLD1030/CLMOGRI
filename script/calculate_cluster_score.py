# 计算聚类评分
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn import metrics

from Utils.excel_util import create_workbook, create_sheet, init_table_head, append_row


def getNClusters(adata, n_cluster, range_min=0, range_max=3, max_steps=40, method='louvain', key_added=None):
    """
    Function will test different settings of louvain to obtain the target number of clusters.
    adapted from the function from the Pinello lab. See: https://github.com/pinellolab/scATAC-benchmarking

    It can get cluster for both louvain and leiden.
    You can specify the obs variable name as key_added.
    """
    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    while this_step < max_steps:
        print('step ' + str(this_step))
        this_resolution = this_min + ((this_max - this_min) / 2)

        if (method == 'louvain') and (key_added == None):
            sc.tl.louvain(adata, resolution=this_resolution)
        elif method == 'louvain' and isinstance(key_added, str):
            sc.tl.louvain(adata, resolution=this_resolution, key_added=key_added)
        elif (method == 'leiden') and (key_added == None):
            sc.tl.leiden(adata, resolution=this_resolution)
        else:
            sc.tl.leiden(adata, resolution=this_resolution, key_added=key_added)

        if key_added == None:
            this_clusters = adata.obs[method].nunique()
        else:
            this_clusters = adata.obs[key_added].nunique()

        print('got ' + str(this_clusters) + ' at resolution ' + str(this_resolution))

        if this_clusters > n_cluster:
            this_max = this_resolution
        elif this_clusters < n_cluster:
            this_min = this_resolution
        elif this_clusters == n_cluster:
            break
            # return(this_resolution, adata)
        else:
            print('Cannot find the number of clusters')
            print('Clustering solution from last iteration is used:' + str(this_clusters) + ' at resolution ' + str(
                this_resolution))

        this_step += 1
# adata = pd.read_csv('../output/GM12878/chr1_embedding.csv', header=None, index_col=0)
# print(adata)
cluster = "chr1"
uni = ["ENSG","chr"]
adata = sc.read(f'../output/GM12878/{cluster}_embedding.csv', header=None, index_col=0)
adata_metapath = sc.read(f'../output/GM12878_metapath/{cluster}_embedding.csv', header=None, index_col=0)

# label = adata.obs.index
label = adata.obs.index
label_metapath = adata_metapath.obs.index
label  = pd.DataFrame(label)
label_metapath  = pd.DataFrame(label_metapath)

# print(label);exit()

# label = index_df = pd.DataFrame(sample_indices, columns=['index'])
# print(label);exit()
label_cluster = label.copy()
label_cluster_metapath = label_metapath.copy()
# print(label_cluster[0]);exit()
label_array=[]
for idl,l in enumerate(label_cluster[0]):
    for i,idx in enumerate(uni):
        if idx in l:
            label_cluster.loc[idl,0] = str(i)
for idl,l in enumerate(label_cluster_metapath[0]):
    for i,idx in enumerate(uni):
        if idx in l:
            label_cluster_metapath.loc[idl,0] = str(i)
sc.pp.neighbors(adata,n_neighbors=30)
sc.pp.neighbors(adata_metapath,n_neighbors=30)

getNClusters(adata,method="louvain",n_cluster=2)
getNClusters(adata,method="leiden",n_cluster=2)

getNClusters(adata_metapath,method="louvain",n_cluster=2)
getNClusters(adata_metapath,method="leiden",n_cluster=2)
score_NMI_specific_louvain = metrics.normalized_mutual_info_score(label_cluster.to_numpy().ravel(),adata.obs['louvain'].to_numpy())
score_NMI_specific_leiden = metrics.normalized_mutual_info_score(label_cluster.to_numpy().ravel(),adata.obs['leiden'].to_numpy())
score_vscore_specific_leiden = metrics.v_measure_score(label_cluster.to_numpy().ravel(),adata.obs['leiden'].to_numpy())
score_vscore_specific_louvain = metrics.v_measure_score(label_cluster.to_numpy().ravel(),adata.obs['louvain'].to_numpy())
score_ARI_specific_leiden = metrics.adjusted_rand_score(label_cluster.to_numpy().ravel(),adata.obs['leiden'].to_numpy())
score_ARI_specific_louvain = metrics.adjusted_rand_score(label_cluster.to_numpy().ravel(),adata.obs['louvain'].to_numpy())

score_NMI_specific_louvain_metapath = metrics.normalized_mutual_info_score(label_cluster_metapath.to_numpy().ravel(),adata_metapath.obs['louvain'].to_numpy())
score_NMI_specific_leiden_metapath = metrics.normalized_mutual_info_score(label_cluster_metapath.to_numpy().ravel(),adata_metapath.obs['leiden'].to_numpy())
score_vscore_specific_leiden_metapath = metrics.v_measure_score(label_cluster_metapath.to_numpy().ravel(),adata_metapath.obs['leiden'].to_numpy())
score_vscore_specific_louvain_metapath = metrics.v_measure_score(label_cluster_metapath.to_numpy().ravel(),adata_metapath.obs['louvain'].to_numpy())
score_ARI_specific_leiden_metapath = metrics.adjusted_rand_score(label_cluster_metapath.to_numpy().ravel(),adata_metapath.obs['leiden'].to_numpy())
score_ARI_specific_louvain_metapath = metrics.adjusted_rand_score(label_cluster_metapath.to_numpy().ravel(),adata_metapath.obs['louvain'].to_numpy())


workbook_ = create_workbook(f"{cluster}_cluster_score_result.xlsx")
worksheet_ = create_sheet(cluster,workbook_)
sheet_headers = ["embedding_type","NMI_louvain","NMI_leiden","vscore_louvain","vscore_leiden","ARI_louvain","ARI_leiden"]
worksheet_ = init_table_head(worksheet_,sheet_headers)
row_node=["node2vec",score_NMI_specific_louvain,score_NMI_specific_leiden,score_vscore_specific_louvain,score_vscore_specific_leiden,score_ARI_specific_louvain,score_ARI_specific_leiden]
row_meta=["metapath2vec",score_NMI_specific_louvain_metapath,score_NMI_specific_leiden_metapath,score_vscore_specific_louvain_metapath,score_vscore_specific_leiden_metapath,score_ARI_specific_louvain_metapath,score_ARI_specific_leiden_metapath]
append_row(worksheet_,row_node)
append_row(worksheet_,row_meta)
workbook_.save(f"{cluster}_cluster_score_result.xlsx")
# sc.pl.umap(adata, color='louvain')
# adata = sc.AnnData(data_specific.transpose())
# sc.pp.neighbors(adata,n_neighbors=30)
# getNClusters(adata,n_cluster=10,method='louvain')
# getNClusters(adata,n_cluster=10,method='leiden')