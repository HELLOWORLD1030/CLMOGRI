# 文件工具类
import os


def create_folder_if_not_exists(folder_path):
    # 验证文件夹是否存在
    if not os.path.exists(folder_path):
        # 如果不存在，创建文件夹
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")
