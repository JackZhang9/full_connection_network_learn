# _*_ Author:JackZhang9
# -*_ Time:20230220
from sklearn.datasets import load_iris
import torch
import torch.nn as nn
import torch.optim as optim

'''鸢尾花数据集全连接神经网络'''
torch.random.seed()
class network(nn.Module):
    '''前向传播过程'''
    def __init__(self):
        super(network, self).__init__()
        '''使用nn.Parameter管理需要更新的参数'''
        self.w1=nn.Parameter(torch.randn((4,10)))
        self.b1=nn.Parameter(torch.randn((10,)))

        self.w2=nn.Parameter(torch.randn((10,20)))
        self.b2=nn.Parameter(torch.randn((20,)))

        self.w3 = nn.Parameter(torch.randn((20, 3)))
        self.b3 = nn.Parameter(torch.randn((3,)))

        # '''第一层隐藏层，(4,10)矩阵，'''
        # self.w1=torch.randn((4,10),requires_grad=True)
        # self.b1=torch.randn((10,),requires_grad=True)
        # '''第二层隐藏层，(4,20)矩阵'''
        # self.w2=torch.randn((10,20),requires_grad=True)
        # self.b2=torch.randn((20,),requires_grad=True)
        # '''第三层隐藏层，(20,3)矩阵'''
        # self.w3 = torch.randn((20, 3),requires_grad=True)
        # self.b3 = torch.randn((3,),requires_grad=True)

    def forward(self,x):
        '''实现当前模块的前向传播过程，x可以是多个参数'''
        z=torch.matmul(x,self.w1)+self.b1
        z=torch.sigmoid(z)

        z = torch.matmul(z, self.w2) + self.b2
        z = torch.sigmoid(z)

        z = torch.matmul(z, self.w3) + self.b3
        return z

def pro():
    '''加载鸢尾花数据'''
    X, Y = load_iris(return_X_y=True)
    # print(X.shape) # (150,4)
    # print(Y.shape) # (150,)
    '''使用torch.from_numpy()把numpy类型数据转化为torch能处理的tensor，并修改数据精度类型'''
    X=torch.from_numpy(X).float()
    Y=torch.from_numpy(Y).long()
    '''实例化一个network对象'''
    net = network()
    loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    opt=optim.SGD(params=net.parameters(),lr=0.01)
    '''传入全部数据，训练2000轮'''
    for i in range(1500):
        y_pre = net(X)
        loss_ = loss(y_pre, Y)
        print('第{}轮,loss={}'.format(i,loss_))

        '''反向传播'''
        loss_.backward()
        opt.step()
        # print('w1 grad=', net.w1.grad)
        # net.w1.data = net.w1.data - 0.01 * net.w1.grad
        # print(net.w1)
    y_p=net(X)
    idx=torch.argmax(y_p,dim=1)
    print(idx)

if __name__ == '__main__':
    pro()




