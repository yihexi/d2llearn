import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_data(filename, batch_size=10):
    print("Loading data...")

    # 读取CSV文件
    dataframe = pd.read_csv(filename)

    # 将数据转换为张量
    features = torch.tensor(dataframe.drop('y', axis=1).values, dtype=torch.float32)
    labels = torch.tensor(dataframe['y'].values, dtype=torch.float32)

    # plot
    # import matplotlib.pyplot as plt
    # plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
    # plt.show()

    labels = labels.reshape(-1, 1)
    print("features shape ", features.shape)
    print("labels shape: ", labels.shape)

    # 创建dataset 和 dataloader
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


if __name__ == "__main__":
    loader = load_data("labeled_data.csv")



    

