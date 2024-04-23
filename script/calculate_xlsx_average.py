# 计算实验结果文件的平均值

import os

import pandas as pd
parent_dir = os.path.dirname(os.getcwd())
# 如果当前目录已经是根目录，那么父目录就是当前目录本身
if os.path.ismount(os.getcwd()):
    # 如果是根目录，父目录就是当前目录
    parent_dir = os.getcwd()
ROOT = parent_dir
excel_file = os.path.join(ROOT, 'result', 'result_258652_19.xlsx')

cluster_list = []
# for i in range(22):
#     cluster_list.append(f"chr{i + 1}")
# cluster_list.append("chrX")
# cluster_list = ["cluster1", "cluster2", "cluster3", "cluster6", "cluster7", "cluster8", "cluster9", "cluster10"]
cluster_list = ["hsc", "cmp", "gmp"]
row_list =[]
for index,cluster in enumerate(cluster_list):
    df =  pd.read_excel(excel_file, sheet_name=cluster, engine='openpyxl')

    test_acc = df["test_acc"].mean().round(3)
    test_aupr = df["test_aupr"].mean().round(3)
    test_auroc = df["test_auroc"].mean().round(3)
    test_f1 = df["test_f1"].mean().round(3)
    test_acc_var = df["test_acc"].var()
    test_aupr_var = df["test_aupr"].var()
    test_auroc_var = df["test_auroc"].var()
    test_f1_var = df["test_f1"].var()

    row = [index+1,cluster, test_acc, test_aupr,test_auroc,test_f1,test_acc_var, test_aupr_var,test_auroc_var,test_f1_var]
    row_list.append(row)
    # print(test_acc, test_aupr,test_auroc,cluster)
# print(row_list)
df = pd.DataFrame(row_list, columns=['序号', '聚类', 'acc','aupr','auroc',"f1", 'acc_方差','aupr_方差','auroc_方差',"f1_方差"])
print(df)
df.to_excel(os.path.join(ROOT, 'result', 'result_258652_19_mean.xlsx'),index=False)