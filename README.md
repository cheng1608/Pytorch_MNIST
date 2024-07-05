>这是WHU大一小学期的一个任务，本人水平有限只会无脑调包，不考虑模型效率，只求能运行出结果
# 1 导包
使用PyTorch搭建神经网络，使用matplotlib生成样例，数据集来自MNIST
```python
import torch  
from torch.utils.data import DataLoader  
from torchvision import transforms  
from torchvision.datasets import MNIST  
import matplotlib.pyplot as plt 
```

# 2 获取数据集
```py
def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)
```
# 3 网络架构
使用三个全连接层进行线性处理
一个Dropout层丢弃部分权重防止过拟合
激活函数使用ReLU
输出函数使用log_softmax，这是softmax的对数版本，在数值计算中更稳定
```py
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.output4 = torch.nn.Linear(32, 10)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.nn.functional.log_softmax(self.output4(x), dim=1)
        return x
```
# 4 评估
使用测试集评估预测的准确率
```py
def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net.forward(x.view(-1, 28 * 28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total
```
# 5 训练函数
选用负对数似然损失，训练五轮
选取测试集中的四张图片展示
```py
def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()

    print("初始准确率:", evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(5):
        for (x, y) in train_data:
            net.zero_grad()
            output = net.forward(x.view(-1, 28 * 28))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward() 
            optimizer.step()

        accuracy = evaluate(test_data, net)
        print("第", epoch + 1, "轮训练 ", "准确率:", accuracy)

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for n, (x, _) in enumerate(test_data):
        if n >= 4:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28)))
        row = n // 2
        col = n % 2
        axs[row, col].imshow(x[0].view(28, 28))
        axs[row, col].set_title("Prediction: " + str(int(predict)))

    plt.tight_layout()
    plt.show()
```

