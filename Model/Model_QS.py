import torch
from torch.nn.modules.module import Module
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from torch.nn.parameter import Parameter
import math
import numpy as np
from torch.autograd import Variable


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        # Calculate cosine similarity between anchor and positive/negatives
        pos_similarity = torch.cosine_similarity(anchor, positive, dim=-1) / self.temperature
        anchor = anchor.unsqueeze(1)
        neg_similarities = torch.cosine_similarity(anchor.unsqueeze(1), negatives, dim=-1) / self.temperature

        # Calculate contrastive term for negatives
        exp_neg_similarities = torch.exp(neg_similarities)
        neg_contrastive = torch.log(exp_neg_similarities.sum(dim=-1))

        # Calculate InfoNCE loss
        loss = -pos_similarity + neg_contrastive
        return loss.mean()



class MultiHeadAttention(nn.Module):
    def __init__(self, input,hidden, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.input = input

        self.query_projection = nn.Linear(self.input, hidden)
        self.key_projection = nn.Linear(self.input, hidden)
        self.value_projection = nn.Linear(self.input, hidden)
        self.output_projection = nn.Linear(hidden, hidden)

    def forward(self, query, key, value, mask=None):
        batch_size, _ = query.size()
        seq_length = self.hidden
        # Project queries, keys, and values to multi-head dimensions
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        # Reshape queries, keys, and values for multi-head attention
        query = query.view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)

        # Compute attention scores and apply mask
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.hidden, dtype=torch.float32))

        if mask is not None:
            mask = scores.sum(0).squeeze(0) + mask
            #scores = scores.masked_fill(mask == 0, -1e9)

        # Compute attention weights
        #attention_weights = F.softmax(scores, dim=-1)
        attention_weights = scores

        # Apply attention weights to values
        attention_output = torch.matmul(attention_weights, value)
        # Reshape attention output and project it back to original dimension
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length)
        attention_output = self.output_projection(attention_output)
        if mask is not None:
            return attention_output, attention_weights,mask
        else:
            return attention_output, attention_weights


class QS_Attention(nn.Module):
    def __init__(self,input,output,heads):
        super(QS_Attention, self).__init__()
        self.value_projection = nn.Linear(input,output)
        self.seq_length = output
        self.num_heads = heads
        self.output_projection = nn.Linear(output, output)
    def forward(self, qk, value, mask=None):
        batch_size, _ = value.size()
        seq_length = self.seq_length
        # Project queries, keys, and values to multi-head dimensions
        value = self.value_projection(value)
        value = value.view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)

        # Compute attention scores and apply mask


        if mask is not None:
            mask = qk.sum(0).squeeze(0) + mask
            #scores = scores.masked_fill(mask == 0, -1e9)

        # Compute attention weights
        #attention_weights = F.softmax(scores, dim=-1)
        attention_weights = qk

        # Apply attention weights to values
        attention_output = torch.matmul(attention_weights, value)
        # Reshape attention output and project it back to original dimension
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length)
        attention_output = self.output_projection(attention_output)
        if mask is not None:
            return attention_output, attention_weights,mask
        else:
            return attention_output, attention_weights

class Inference(nn.Module):
    def __init__(self,input,hidden,output):
        super(Inference, self).__init__()
        self.output = output
        self.n_class = output
        self.Linear1 = nn.Linear(input,hidden,bias=True)
        self.ReLU = nn.ReLU()
        self.inference_get_logits = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,output)
        )


    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z

    def soft_cross_entropy(self, y_hat, y_soft, weight=None):
        if weight is None:
            loss = - torch.sum(torch.mul(torch.log_softmax(y_hat, dim=-1), y_soft)) / self.n_class
        else:
            loss = - torch.sum(torch.mul(weight, torch.mul(torch.log_softmax(y_hat, dim=-1), y_soft))) / self.n_class
        return loss

    def get_logits(self,x):
        num_layers = len(self.inference_get_logits)
        for i,layer in enumerate(self.inference_get_logits):
            if i == num_layers - 1:
                #x = layer(x,temperature)
                x = layer(x)
            else:
                x = layer(x)
        return x
    def forward(self,x):
        x = self.ReLU(self.Linear1(x))
        logits = self.inference_get_logits(x)
        return logits

class Decoder(nn.Module):
    def __init__(self,input,hidden,output_class,node_num,output_embed):
        super(Decoder, self).__init__()
        self.hidden = hidden
        self.output = node_num
        self.output_class = output_class
        self.output_embed = output_embed
        self.weight = nn.Linear(input, self.hidden, bias=True)
        self.weight2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden,self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden,output_class),
        )
        self.weight3 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden,self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden,self.output_embed),
        )
        self.weight4 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden,self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden,self.output),
            nn.ReLU(),
        )
    def forward(self,x):
        q = self.weight(x)
        output_embed = self.weight3(q)
        output_class = self.weight2(q)
        output_pred = self.weight4(q)
        output_class = nn.Softmax(1)(output_class)
        output_embed = nn.Sigmoid()(output_embed)
        output_pred = nn.Softmax(1)(output_pred)
        return output_embed,output_class,output_pred

class Model_QS(nn.Module):
    def __init__(self,input,output,hidden,temperature):
        super(Model_QS,self).__init__()
        self.Encoder = Inference(input,hidden,output)
        self.Multiattention = MultiHeadAttention(output,output,1)
        self.QSattention = QS_Attention(output,output,1)
        self.Decoder = Decoder(output,hidden,2,2,output)
        self.Loss_Info = InfoNCELoss(temperature=temperature)



    def forward(self,Node_A,Node_B):
        X_sliced_A = Node_A[:, :-2]
        X_sliced_B = Node_B[:, :-2]
        self.normalized_probs = X_sliced_A.mean(dim=0) / X_sliced_A.mean(dim=0).sum(dim=0, keepdim=True)
        sampled_samples = self.normalized_probs*(X_sliced_B+X_sliced_A)/2

        x_A = self.Encoder(Node_A)
        x_B = self.Encoder(Node_B)

        output_A,weight_A = self.Multiattention(x_A,x_A,x_A)
        output_B,weight_B = self.QSattention(weight_A,x_B)

        output_embed,output_class,output_pred = self.Decoder(output_B)


        loss_info = self.Loss_Info(output_A,output_B,sampled_samples)

        return output_embed,output_class,output_pred,loss_info

    def fit(self, train_loader, val_loader, epochs, learning_rate, device):
        """
        训练模型

        :param train_loader: 训练数据的DataLoader
        :param val_loader: 验证数据的DataLoader
        :param epochs: 训练的轮数
        :param learning_rate: 学习率
        :param device: 设备（'cpu' 或 'cuda'）
        """
        self.to(device)  # 将模型移动到指定设备
        self.train()  # 设置模型为训练模式

        # 初始化优化器
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # 训练循环
        for epoch in range(epochs):
            for batch_idx, (Node_A, Node_B) in enumerate(train_loader):
                # 将数据移动到指定设备
                Node_A, Node_B = Node_A.to(device), Node_B.to(device)

                # 清除之前的梯度
                optimizer.zero_grad()

                # 前向传播
                output_embed, output_class, output_pred, loss_info = self.forward(Node_A, Node_B)

                # 计算损失
                loss = loss_info  # 这里假设你只关心InfoNCELoss

                # 反向传播和参数更新
                loss.backward()
                optimizer.step()

                # 打印进度
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')

            # 验证循环（可选）
            # with torch.no_grad():
            #     self.eval()  # 设置模型为评估模式
            #     for Node_A, Node_B in val_loader:
            #         Node_A, Node_B = Node_A.to(device), Node_B.to(device)
            #         # ...（评估代码）
            #         print(f'Validation Loss: {validation_loss.item()}')