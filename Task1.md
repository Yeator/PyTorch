# PyTorch的基本概念
## 1. 什么是PyTorch，为什么选择PyTorch？
PyTorch是一个机遇Python的库，用来提供一个具有灵活性的深度学习开发平台，其工作流程非常接近Python的科学计算库——numpy。

选择PyTorch的原因有以下几点：
① 易于使用的API：就像Python一样简单。
② Python的支持：PyTorch可以顺利地与Python数据科学栈集成，它非常类似于numpy，甚至注意不到它们的差别。
③ 动态计算图：取代了具有特定功能的预定义图形，PyTorch为我们提供了一个框架，以便可以在运行时构建计算图，甚至在运行时更改它们。在不知道创建神经网络需要多少内存的情况下这非常有价值。
④其余优点：多gpu支持，自定义数据加载器和简化的预处理器。
## 2. PyTorch的安装
环境：win10 + Python 3.6 + conda 3.5 + Pytorch
## 3. 配置Python环境
[使用anaconda配置]https://blog.csdn.net/benben513624/article/details/80066136
## 4. 准备Python管理器
与下一步结合一起看。
## 5. 通过命令行安装PyTorch
[PyTorch安装](https://redstonewill.com/1948/)
[使用pycharm验证](https://zhuanlan.zhihu.com/p/35255076)
## 6. PyTorch基础概念
张量：Tensorflow中数据的核心单元就是Tensor。张量包含了一个数据集合，这个数据集合就是原始值变形而来的，它可以是一个任何维度的数据。tensor的rank就是其维度。
Pytorch对张量的操作：https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch/
Pytorch中的数学运算与Python中的numpy库类似
Autograd模块：Pytorch中的自动微分模块
Optim模块：Pytorch中的优化算法模块
神经网络模块：torch.nn
## 7. 通用代码实现流程（实现一个深度学习的代码流程）
通用流程：
① 设置训练参数
② 加载训练集和测试集
③ 搭建网络
④ 选择优化器
⑤ 制定训练过程和测试过程
⑥ 主函数执行
[PyTorch: CNN实战MNIST手写数字识别](https://blog.csdn.net/m0_37306360/article/details/79311501)
```
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# 训练参数设置
kernel_size = 5
batch_size = 64
epoch_num = 10

# 下载MNIST数据集，并加载
train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 网络搭建
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=kernel_size)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = func.relu(self.mp(self.conv1(x)))
        x = func.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)
        x = self.fc(x)
        return func.log_softmax(x)

# 生成实例，选择优化器
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 训练过程
def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = func.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# 测试过程
def test():
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += func.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# 主函数
if __name__=="__main__":
    for epoch in range(1, epoch_num):
        train(epoch)
        test()
```
