#使用metapath方法对GM12878数据集进行embedding
# 前置工作:"walkpath_metapath_GM12878.py"
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from Data_Datasets import DataReader,Metapath2vecDataset
from Model.Model_Metapath2vec import SkipGramModel


class Metapath2VecTrainer:
    def __init__(self, file, min_count, window_size, batch_size, output_file, dim, iterations, initial_lr):
        self.data = DataReader(file, min_count)
        dataset = Metapath2vecDataset(self.data, window_size)
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=True, num_workers=0, collate_fn=dataset.collate)

        self.output_file_name = output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = dim
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr #learning rate
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):

        for iteration in range(self.iterations):
            print("\n\n\nIteration: " + str(iteration + 1))
            optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    scheduler.step()
                    optimizer.zero_grad()
                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 500 == 0:
                        print(" Loss: " + str(running_loss))

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)
min_count = 5   #单词频率截断
window_size = 5  #窗口大小
batch_size = 64  #批大小
dim = 128  # embedding维度
iterations = 5  # 循环次数
initial_lr = 0.001  #learning rate
cluster_list = []
for i in range(2,22):
    cluster_list.append(f"chr{i + 1}")
cluster_list.append("chrX")
# cluster_list = ["chr1"]
for cluster in cluster_list:
    file = f'../Data/embedding/GM12878/metapath2vec/walk_path/{cluster}.csv'# walk_path路径
    output_file = f'../Data/embedding/GM12878/metapath2vec/metapath2vec_{cluster}_embedding.csv'  #embedding输出路径
    m2v = Metapath2VecTrainer(file, min_count, window_size, batch_size, output_file, dim, iterations, initial_lr)
    m2v.train()

