import torch
import os
from utils.dataset import MyDataset
from torch.utils.data import DataLoader, random_split
from snet import SpectrumNet


def train_net(val_percent=0.1, batch_size=1, save_cp=True):
    # 数据加载
    dataset = MyDataset(dir_data)

    # 数据划分
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])  # 神器
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    # Optimizer Parameters
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

    # Loss Function
    criterion = torch.nn.BCELoss(size_average=True)

    for epoch in range(100):
        net.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            y_pre = net(inputs)
            loss = criterion(y_pre, labels)
            print(epoch, i, loss.item())  # 请使用 logging.info 高级技术

            # 梯度下降
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 保存权重文件
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
            except OSError:
                pass
            # 保存 weight 和 bias
            torch.save(net.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')


def val_net():
    pass


dir_data = ''
dir_checkpoint = ''

if __name__ == '__main__':
    net = SpectrumNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    train_net(batch_size=1, val_percent=0.1)
