import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as Data
import glob
from matplotlib import pyplot as plt
import PIL.Image as Image
import os

TARGET_CLASS = 4
MODEL_NAME = "..\\models\\brokenline.pth"


# 加载数据
def load_data(batch_size=15, display=False):
    t = [transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.RandomRotation(40),
         transforms.RandomRotation(20)]
    trans = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomApply(t, p=0.5),
        transforms.RandomResizedCrop(28),
        transforms.ToTensor()
    ])
    train_data = torchvision.datasets.ImageFolder(root='..\\train_data\\brokenline',
                                                  transform=trans)

    print(len(train_data))

    # 小批量数目
    train_iter = torch.utils.data.DataLoader(train_data,
                                             batch_size=batch_size,
                                             shuffle=True)
    # num_workers=0,不开启多线程读取。
    # test_iter = torch.utils.data.DataLoader(mnist_test,
    #                                         batch_size=batch_size,
    #                                         shuffle=False)

    # 显示10张图
    if display:
        features, labels = iter(train_iter).next()
        # common.show_imgs(features[0:10], labels[0:10], 5, True)

    return train_iter


class myLeNet(nn.Module):
    def __init__(self):
        super(myLeNet, self).__init__()
        self.conv = nn.Sequential(
            # in_channels, out_channels, kernel_size
            # 28*28->24*24
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            # kernal_size, stride
            # 24*24->12*12
            nn.MaxPool2d(2, 2),
            # 12*12->8*8
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            # 8*8->4*4
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, TARGET_CLASS),
        )

    def forward(self, x):
        x = self.conv(x)
        # flatten features as input
        x1 = x.view(x.shape[0], -1)
        x = self.fc(x1)

        return x


def test_FashionNet():
    if not os.path.exists(MODEL_NAME):
        print("模型文件不存在")

    # 加载网络..
    model = myLeNet()
    model.load_state_dict(torch.load(MODEL_NAME))

    imgpath = ".\\brokenline_test\\"
    for pathfile in glob.glob(imgpath+"*.jpg"):
        filename = os.path.basename(pathfile)
        img = Image.open(pathfile)
        if not img.width == 28 or not img.height == 28:
            continue
        feature = transforms.ToTensor()(img).unsqueeze(dim=0)
        _pred = model(feature).squeeze(dim=0)
        pred_prob = nn.Softmax(dim=0)(_pred).unsqueeze(dim=0)
        _, index = torch.max(pred_prob.data, 1)

        newfile = imgpath + str(index.item()) + filename
        img.save(newfile)
    print("done")


def train_FashionData():
    lr = 0.01
    epoch = 200
    batch_size = 50

    train_iter = load_data(batch_size)
    use_gpu = torch.cuda.is_available()

    model = myLeNet()
    if os.path.exists(MODEL_NAME):
        print("加载模型文件")
        # 加载网络..
        model.load_state_dict(torch.load(MODEL_NAME))

    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    # 设置了momentum的SGD比未设置的效果好多了。
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8)

    loss_data = []
    for e in range(epoch):
        accuracy = 0
        loss_total = 0
        for step, (features, label) in enumerate(train_iter):
            # features = features.view(features.size(0), -1)

            if use_gpu:
                features = features.cuda()
                label = label.cuda

            y_hat = model(features)
            loss = criterion(y_hat, label)
            loss_total += loss.item()
            # print("batch loss: {0}".format(loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 精确度
            # 1 表示每行的最大值
            _, y_pred = torch.max(y_hat, 1)
            # 每批次的准确度。因为都是0和1，可利用mean求每次的准确度
            accuracy += (y_pred == label).float().mean()

        accuracy = accuracy / (step + 1)
        loss_data.append(loss_total/(step+1))
        if e % 10 == 0:
            print("epoch {0}: accuracy={1}, batch_average_loss={2}".format(e, accuracy, loss_total/(step+1)))

    print("Final: accuracy={0}, loss={1}".format(accuracy, loss_total))

    # 保存模型
    torch.save(model.state_dict(), MODEL_NAME)

    # show train_loss curve
    x = torch.arange(0, epoch, 1)
    plt.title('train loss')
    plt.plot(x, loss_data)
    plt.show()


def patchCheck(model, img):
    if not img.shape[0] == 28 or not img.shape[1] == 28:
        return -1, 0
    feature = transforms.ToTensor()(img).unsqueeze(dim=0)
    _pred = model(feature).squeeze(dim=0)
    pred_prob = nn.Softmax(dim=0)(_pred).unsqueeze(dim=0)
    confidence, index = torch.max(pred_prob.data, 1)

    return index.item(), confidence.item()


def main():
    # load_data(batch_size=20, display=True)
    train_FashionData()
    # test_FashionNet()


if __name__ == '__main__':
    main()

