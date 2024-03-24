import torch
from torch import nn
import data_loader
import torch.onnx
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    # 创建一个SummaryWriter对象，指定写入的目录
    writer = SummaryWriter('logs')

    # 加载数据
    batch_size = 10
    train_loader = data_loader.load_data("labeled_data.csv", batch_size=batch_size)
    print("Data loaded successfully")

    # from d2l import torch as d2l
    # true_w = torch.tensor([2, -3.4])
    # true_b = 4.2
    # features, labels = d2l.synthetic_data(true_w, true_b, 1000)
    # train_loader = d2l.load_array((features, labels), batch_size)

    # 定义模型
    model = nn.Sequential(nn.Linear(2, 1))
    model[0].weight.data.normal_(0, 0.01)
    model[0].bias.data.fill_(0)

    # 定义损失函数和优化器
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)

    # 训练模型
    num_epochs = 100
    i = 0
    for epoch in range(num_epochs):
        running_loss = 0.0

        for j, (X, y) in enumerate(train_loader):
            l = loss(model(X), y)
            optimizer.zero_grad()

            l.backward()
            optimizer.step()

            running_loss += l.item()

        # 每个epoch 打印一次损失
        print("[%d] loss: %.3f" % (epoch + 1, running_loss / batch_size))
        print("weight: ", model[0].weight.data)
        print("bias: ", model[0].bias.data)

        writer.add_scalar("training_loss", running_loss / batch_size, epoch * len(train_loader) + i)
        i += 1

        # if 损失小于一个阈值，结束训练，保存模型
        if running_loss / batch_size < 1:
            dummy_input_half = torch.randn(100, 1, 2, dtype=torch.float16)
            model_half = model.half()
            model_half.eval()

            torch.onnx.export(model_half,
                              dummy_input_half,
                              "linear2d_onnx_model.onnx",
                              output_names=["y"],
                              input_names=["X"],
                              export_params=True,
                              do_constant_folding=True,
                              dynamic_axes={"X": {0: "batch_size"}, "y": {0: "batch_size"}})
            break

    print('Finished Training')
    writer.close()


