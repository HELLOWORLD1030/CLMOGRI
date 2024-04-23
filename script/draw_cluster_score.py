import matplotlib.pyplot as plt
import pandas as pd

# 读取Excel文件中的数据
data = pd.read_excel('chr20_cluster_score_result.xlsx', header=0)

# 创建一个新的DataFrame来存储转换后的数据
df = pd.DataFrame({
    'embedding_type': data['embedding_type'].iloc[0],
    'NMI_louvain': data['NMI_louvain'],
    'NMI_leiden': data['NMI_leiden'],
    'ARI_louvain': data['ARI_louvain'],
    'ARI_leiden': data['ARI_leiden'],
    'VSCORE_louvain': data['vscore_louvain'],
    'VSCORE_leiden': data['vscore_leiden']
})

# 创建一个图表和两个子图
fig, axs = plt.subplots(1, 2, figsize=(12, 8))

# 绘制NMI得分的条形图
axs[0].bar(df['embedding_type'], df['NMI_louvain'], label='NMI Louvain', color='blue')
axs[0].bar(df['embedding_type'], df['NMI_leiden'], bottom=df['NMI_louvain'], label='NMI Leiden', color='red')
axs[0].set_title('NMI Scores')
axs[0].set_xlabel('Embedding Type')
axs[0].set_ylabel('Score')
axs[0].legend()

# 绘制ARI和VSCORE得分的条形图
axs[1].bar(df['embedding_type'], df['ARI_louvain'], label='ARI Louvain', color='green')
axs[1].bar(df['embedding_type'], df['ARI_leiden'], bottom=df['ARI_louvain'], label='ARI Leiden', color='orange')
axs[1].bar(df['embedding_type'], df['VSCORE_louvain'], label='VSCORE Louvain', color='purple', alpha=0.5)
axs[1].bar(df['embedding_type'], df['VSCORE_leiden'], bottom=df['ARI_louvain'] + df['VSCORE_louvain'], label='VSCORE Leiden', color='pink', alpha=0.5)
axs[1].set_title('ARI and VSCORE Scores')
axs[1].set_xlabel('Embedding Type')
axs[1].set_ylabel('Score')
axs[1].legend()

# 调整子图间距
plt.tight_layout()

# 显示图表
plt.show()