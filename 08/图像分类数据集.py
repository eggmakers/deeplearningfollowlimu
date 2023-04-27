import torch    
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()


#通过框架自带的函数下载数据
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root = "data", train=True, transform=trans, download=False)
mnist_test = torchvision.datasets.FashionMNIST(root = "data", train=False, transform=trans, download=False)

print(len(mnist_train), '\n', len(mnist_test))
print(mnist_train[0][0].shape)#灰度图


#两个可视化数据集的函数
def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, title = None, scale = 1.5):
    """Plot a list of images"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize = figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            #图片张量
            ax.imshow(img.numpy())
        else:
            #PIL图片
            ax.imshow(img)


#几个样本的图像以及相应的标签
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, title=get_fashion_mnist_labels)
d2l.plt.show()#图片显示


#读一小批量数据
batch_size = 256

def get_dataloader_workers():
    """使用8个进程来读取数据"""
    return 8

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())

timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')


#定义load_data_fashion_mnist函数
def load_data_fashion_mnist(batch_size, resize = None):
    """下载Fashion-MNIST数据集， 然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root = "data", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root = "data", train=False, transform=trans, download=False)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=get_dataloader_workers()))


#测试
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break