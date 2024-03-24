import os
import torch
import numpy as np
from d2l import torch as d2l


def save_data_to_csv(features, labels, filename):
    """保存数据到csv文件"""
    features_np = features.numpy()
    labels_np = labels.numpy()
    np.savetxt(filename, np.concatenate((features_np, labels_np), axis=1), delimiter=",",
                header="x1,x2,y", comments="")


if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)

    # plot
    # d2l.set_figsize()
    # d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
    # d2l.plt.show()

    path = os.getcwd()
    save_data_to_csv(features, labels, path+"/labeled_data.csv", )

    print("features shape: ", features.shape)
    print("labels shape: ", labels.shape)



