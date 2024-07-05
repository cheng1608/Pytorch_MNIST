import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)


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


def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()

    print("initial accuracy:", evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(5):
        for (x, y) in train_data:
            net.zero_grad()
            output = net.forward(x.view(-1, 28 * 28))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()  # 反向传播
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


if __name__ == "__main__":
    main()
