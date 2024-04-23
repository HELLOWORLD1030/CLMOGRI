# 使用QS模型训练GEEK的embedding数据(由Node2vec)
import os.path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import average_precision_score, accuracy_score, roc_curve, auc, f1_score
from Model.Model_QS import *
import dill

from Utils.excel_util import *


class scmti:
    def __init__(self, cluster, save, seed):
        self.cluster = cluster
        parent_dir = os.path.dirname(os.getcwd())
        # 如果当前目录已经是根目录，那么父目录就是当前目录本身
        if os.path.ismount(os.getcwd()):
            # 如果是根目录，父目录就是当前目录
            parent_dir = os.getcwd()
        self.ROOT = parent_dir
        self.DATA_FOLDER = os.path.join(self.ROOT, "Data")
        self.OUTPUT_FOLDER = os.path.join(self.ROOT, "output")

        self.GM12878_FOLDER = os.path.join(self.DATA_FOLDER, "A unified data", "GM12878_intra", "GM12878")
        self.GM12878_OUTPUT_FOLDER = os.path.join(self.OUTPUT_FOLDER, "GM12878")
        self.run_count = 21  # 运行次数
        # self.set_run_count()
        self.cluster = cluster
        self.device = torch.device("cuda:0")
        self.path_embeddings = os.path.join(self.DATA_FOLDER,"embedding","GM12878","node2vec", f'node2vec_{self.cluster}_embedding.csv')
        self.path_node = os.path.join(self.GM12878_FOLDER, f'node')
        self.path_edge = os.path.join(self.GM12878_FOLDER, f'edge')
        # self.path_label = '../Data/scMTNI_sourcedata/scMTNI_sourcedata/Buenrostro_Hematopoiesis/scMTNI_networks'
        self.path_Gene = os.path.join(self.path_node, 'gene_expr', f'{self.cluster}.gene')
        self.path_Regu = os.path.join(self.path_node, 'bin_DNase', f'{self.cluster}.bin.bed.DNase')
        self.path_label = os.path.join(self.path_edge, 'merge', f'{self.cluster}.csv')
        # self.seed = 3277538  # 设置随机种子，可以是任意整数
        self.seed = seed
        self.pearson = 0.7
        self.result_path = f'../Result/result_GM12878_node2vec_{self.seed}_{self.run_count}.xlsx'
        self.pkl_file = f'../Data/dataset_pkl/dataset_train_save_GM12878_node2vec_{self.cluster}.pkl'
        self.save = save  # 1采用本地文件处理，不走数据预处理流程 0走数据预处理流程
        self.workbook = create_workbook(self.result_path)
        self.worksheet = create_sheet(self.cluster, self.workbook)
        self.sheet_headers = ["epoch", "random_seed", "loss_info_nce", "train_acc", "train_aupr", "train_auroc",
                              "train_f1"
            , "test_acc",
                              "test_aupr", "test_auroc", "test_f1"]
        self.worksheet = init_table_head(self.worksheet, self.sheet_headers)

    def initalize_A(self, data_shape):
        num_genes = data_shape
        A = np.ones([num_genes, num_genes]) / (num_genes - 1) + np.random.randn(num_genes * num_genes).reshape(
            [num_genes, num_genes]) * 0.5
        for i in range(len(A)):
            A[i, i] = 0
        return A

    '''
    初始化
    '''

    def xavier_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def _one_minus_A_t(self, adj):
        adj_normalized = Tensor(np.eye(adj.shape[0])) - (adj.transpose(0, 1))
        return adj_normalized

    '''
    数据准备
    '''

    def dataset_preparation(self, path_output, path_label, pearson):
        # [4079 rows x 1 columns]
        # PRMT2_cluster1
        node_Gene = pd.read_csv(self.path_Gene, header=None)  # 基因节点

        # [638 rows x 1 columns]
        # ZNRD1_cluster1
        node_Regu = pd.read_csv(self.path_Regu, header=None)  # 调控子节点
        # [740388 rows x 3 columns]
        edge = pd.read_csv(self.path_label)  # out
        edge.columns = [0, 1, 2]
        ui = pd.read_csv(self.path_embeddings, header=None, index_col=0)  # embedding
        BGGB_embed = ui.copy()  # (11201, 128)
        node_list = ui.index.to_numpy()
        node_num = len(node_list)
        adj_matrix = np.zeros((node_num, node_num))  # 邻接矩阵
        # [220279 rows x 3 columns]
        edges = edge[edge[2] > 0.7]

        edges_d = {}
        # print(edges.to_dict('records'))
        data_A = []
        data_B = []
        data_label = []
        data_class = []
        data_node = []
        data_edge = []
        # edges_d key是调控子和基因拼起来 value是置信度
        for item in (edges.to_dict('records')):
            edges_d[item[0] + item[1]] = item[2]
        # print(edges_d)
        # 设置邻接矩阵
        for idk, row in edges.iterrows():
            node1 = row[0]
            node2 = row[1]
            weight = row[2]
            if node1 in node_list and node2 in node_list:
                node1_index = np.where(node_list == node1)[0][0]
                node2_index = np.where(node_list == node2)[0][0]
                adj_matrix[node1_index, node2_index] = weight
                adj_matrix[node2_index, node1_index] = weight

        adj_df = pd.DataFrame(adj_matrix, index=node_list, columns=node_list)
        embed = BGGB_embed.to_numpy()  # (11201, 128)
        result_embed = np.corrcoef(embed, rowvar=True)  # 计算相关系数
        embed_tri = np.triu(result_embed)  # 处理成上三角矩阵
        for idk in range(embed_tri.shape[0]):
            embed_tri[idk, idk] = 0  # 对角线置0
        location = np.where(embed_tri > pearson)
        print(location)
        location_df = pd.DataFrame(columns=['0', '1'])
        location_df['0'] = BGGB_embed.index[location[0]]
        location_df['1'] = BGGB_embed.index[location[1]]
        location_df['correlation'] = embed_tri[location[0], location[1]]

        for item in location_df.iterrows():
            temp = []
            temp_B = []
            loca = np.where(ui.index == item[1][0])
            # loca_rank = np.where(ranked_matrix.index == item[1][0])
            loca_next = np.where(ui.index == item[1][1])
            if len(loca[0]) != 0 and len(loca_next[0]) != 0:
                embed_vec = ui.iloc[loca[0]].to_numpy()[0]
                # print(embed_vec)

                pre_vec = ui.iloc[loca_next[0]].to_numpy()[0]

                if ui.index[loca[0]][0] in node_Gene.values:  # 基因

                    class_A = list([0])  #
                else:
                    class_A = list([1])  # 调控子
                if ui.index[loca_next[0]][0] in node_Gene.values:  # 基因

                    class_B = list([0])
                else:
                    class_B = list([1])
                if item[1][0] + item[1][1] in edges_d or item[1][1] + item[1][0] in edges_d:
                    edges_temp = list([1])
                else:
                    edges_temp = list([0])
                node_feature = item[1][2]
                node_feature_B = item[1][2]
                temp.extend(embed_vec)
                temp.extend(class_A)
                temp.append(node_feature)
                temp_B.extend(pre_vec)
                temp_B.extend(class_B)
                temp_B.append(node_feature_B)
                data_A.append(temp)
                data_B.append(temp_B)

                data_class.append(class_B)
                data_label.append(embed_vec)
                data_edge.append(edges_temp)

        dataset_embed = TensorDataset(torch.tensor(np.array(data_A)), torch.tensor(np.array(data_label)),
                                      torch.tensor(np.array(data_class)),
                                      torch.tensor(np.array(data_B)), torch.tensor(np.array(data_edge)))
        train_size = int(len(dataset_embed) * 0.8)
        test_size = int(len(dataset_embed) - train_size)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset_embed, [train_size, test_size],
                                                                    generator=torch.Generator().manual_seed(42))
        with open(self.pkl_file, 'wb') as f:
            dill.dump([train_dataset, test_dataset, result_embed, node_num], f)

    def get_prepared_data(self, save=0):
        print(type(save), save)
        # 从源文件开始数据处理逻辑
        if save == 0:
            self.dataset_preparation(self.path_embeddings, self.path_label, self.pearson)
            with open(self.pkl_file, 'rb') as f:
                train_dataset, test_dataset, result_embed, node_num = dill.load(f)
                # print(train_dataset)
                trainLoader = DataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=True)
                testLoader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)
                adj_A_init = self.initalize_A(node_num)
                adj_A_t = self._one_minus_A_t((torch.tensor(adj_A_init)))
                # # print(len(trainLoader))
                return result_embed, trainLoader, testLoader, adj_A_t
        else:
            with open(self.pkl_file, 'rb') as f:
                train_dataset, test_dataset, result_embed, node_num = dill.load(f)
                print(train_dataset)
                trainLoader = DataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=True)
                testLoader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)
                adj_A_init = self.initalize_A(node_num)
                adj_A_t = self._one_minus_A_t((torch.tensor(adj_A_init)))
                # # print(len(trainLoader))
                return result_embed, trainLoader, testLoader, adj_A_t

    def train(self, epochs):
        torch.manual_seed(self.seed)  # 设置 Torch 的随机种子
        torch.cuda.manual_seed(self.seed)  # 设置 CUDA 的随机种子
        torch.backends.cudnn.deterministic = True  # 设置为 True 以确保结果的可复现性
        result_embed, trainLoader, testLoader, adj_A_t = self.get_prepared_data(self.save)
        main_net = Model_QS(130, 128, 64, 1).to(device=self.device)

        main_net.apply(self.xavier_init).to(device=self.device)

        main_net_optimizer = torch.optim.Adam(main_net.parameters(), lr=1e-5)

        main_net_scheduler = torch.optim.lr_scheduler.StepLR(main_net_optimizer, step_size=50,
                                                             gamma=0.99)
        loss_MSE = torch.nn.MSELoss().to(self.device)

        BCE = torch.nn.BCELoss().to(self.device)

        best = 0
        for epoch in range(epochs):
            # for i in tqdm(range(epochs)):
            print("---第{}轮---".format(epoch + 1))
            main_net.train()
            for i, batch in enumerate(trainLoader):
                # dataset_embed = TensorDataset(torch.tensor(np.array(data_A)), torch.tensor(np.array(data_label)),
                #                               torch.tensor(np.array(data_class)),
                #                               torch.tensor(np.array(data_B)), torch.tensor(np.array(data_edge)))
                inputs_A, label_embed, label_class, inputs_B, label_edge = batch
                inputs_A = inputs_A.to(device=self.device)  # 节点a
                inputs_B = inputs_B.to(device=self.device)  # 节点b

                label_embed = label_embed.to(device=self.device)  # embeding的向量
                label_class = label_class.to(device=self.device)  # 节点分类（基因 or 调控子）
                label_edge = label_edge.to(device=self.device)  # ab之间有没有边
                main_net_optimizer.zero_grad()
                output_data, output_class, output_edge, InfoNCE_loss = main_net(inputs_A.to(torch.float32),
                                                                                inputs_B.to(torch.float32))
                MSE_loss = loss_MSE(output_data, label_embed.to(torch.float32).to(device=self.device))
                BCE_loss = BCE(output_class.argmax(-1).unsqueeze(-1).to(torch.float32),
                               label_class.to(torch.float32).to(device=self.device))
                Edge_loss = BCE(output_edge.argmax(-1).unsqueeze(-1).to(torch.float32),
                                label_edge.to(torch.float32).to(device=self.device))
                distance = torch.dist(output_data, label_embed)
                loss = InfoNCE_loss
                loss.backward()
                main_net_optimizer.step()
                main_net_scheduler.step()
            acc_train = accuracy_score(label_edge.to(torch.float32).cpu().detach().numpy(),
                                       output_edge.argmax(-1).cpu().detach().numpy().round())
            aupr_edge_train = average_precision_score(label_edge.to(torch.float32).cpu().detach().numpy(),
                                                      output_edge.argmax(-1).cpu().detach().numpy().round())
            # aupr_node = average_precision_score(label_class.to(torch.float32).cpu().detach().numpy(),
            #                                     output_data.argmax(-1).cpu().detach().numpy().round())
            # distance = torch.dist(output_data, label_embed)
            FPR, TPR, thresholds = roc_curve(label_edge.to(torch.float32).cpu().detach().numpy(),
                                             output_edge.argmax(-1).cpu().detach().numpy().round()
                                             )
            AUROC_train = auc(FPR, TPR)
            loss_train = loss.cpu().detach().numpy()
            f1_train = f1_score(label_edge.to(torch.float32).cpu().detach().numpy(),
                                output_edge.argmax(-1).cpu().detach().numpy().round(), average='binary')
            print(loss_train, AUROC_train, f1_score)
            print('acc:', acc_train, 'Distance:', distance, 'aupr_edge:', aupr_edge_train)
            with torch.no_grad():
                for i, batch in enumerate(testLoader):
                    inputs_A_test, label_embed_test, label_class_test, inputs_B_test, label_edge_test = batch
                    inputs_A_test = inputs_A_test.to(device=self.device)
                    inputs_B_test = inputs_B_test.to(device=self.device)
                    label_embed_test = label_embed_test.to(device=self.device)
                    label_class_test = label_class_test.to(device=self.device)
                    label_edge_test = label_edge_test.to(device=self.device)
                    output_data_test, output_class_test, output_edge_test, InfoNCE_loss_test = main_net(
                        inputs_A_test.to(torch.float32),
                        inputs_B_test.to(torch.float32))
                    loss_test = InfoNCE_loss_test
            acc_test = accuracy_score(label_edge_test.to(torch.float32).cpu().detach().numpy(),
                                      output_edge_test.argmax(-1).cpu().detach().numpy().round())
            aupr_edge_test = average_precision_score(
                label_edge_test.to(torch.float32).cpu().detach().numpy(),
                output_edge_test.argmax(-1).cpu().detach().numpy().round())
            aupr_node_test = average_precision_score(
                label_class_test.to(torch.float32).cpu().detach().numpy(),
                output_data_test.argmax(-1).cpu().detach().numpy().round())
            distance_test = torch.dist(output_data_test, label_embed_test)
            FPR, TPR, thresholds = roc_curve(
                label_edge_test.to(torch.float32).cpu().detach().numpy(),
                output_edge_test.argmax(axis=-1).cpu().detach().numpy().round())
            AUROC_test = auc(FPR, TPR)
            f1_test = f1_score(label_edge_test.to(torch.float32).cpu().detach().numpy(),
                               output_edge_test.argmax(-1).cpu().detach().numpy().round(), average='binary')

            print('test:', loss_test, AUROC_test)
            print('aupr_node:', aupr_node_test, 'acc:', acc_test, 'Distance:', distance_test, 'aupr_edge:',
                  aupr_edge_test)
            if AUROC_test > best:
                torch.save(main_net, f=f'../Output/model/model_CLMOGRI_GM12878_node2vec_{cluster}.pth')

                best = AUROC_test
            print('best:', best)
            row = [str(epoch + 1), self.seed, str(loss_train), str(acc_train)
                , str(aupr_edge_train)
                , str(AUROC_train)
                , str(f1_train)
                , str(acc_test)
                , str(aupr_edge_test)
                , str(AUROC_test)
                , str(f1_test)
                   ]
            append_row(self.worksheet, row)
        self.workbook.save(self.result_path)


if __name__ == "__main__":
    cluster_list = []
    for i in range(22):
        cluster_list.append(f"chr{i + 1}")
    cluster_list.append("chrX")
    # cluster_list = ["chr22"]
    seed_list = [258652]
    for seed in seed_list:
        for cluster in cluster_list:
            sc = scmti(cluster=cluster, save=0, seed=seed)
            sc.train(epochs=15)
