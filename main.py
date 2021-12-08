import torchvision
import torch
from sklearn.naive_bayes import GaussianNB
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

batch_size = 32
train_data = torchvision.datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data = torchvision.datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)


def pretreatment(data, batch_size):
    image = []
    label = []
    for i in range(len(data)):
        x, y = next(iter(data))
        for ii in range(batch_size):
            image.append(x[ii].reshape(1, -1)[0].tolist())
            label.append(y[ii].item())

    image = np.array(image)
    label = np.array(label)
    return image, label


train_image, train_label = pretreatment(train_loader, batch_size)

val_image, val_label = pretreatment(val_loader, batch_size)

classify = GaussianNB().fit(train_image, train_label)

predict_label = classify.predict(val_image)

plt.figure(1)
print("预测结果为：",end=" ")
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(val_image[i].flatten().reshape(28, 28))
    print(predict_label[i], end=" ")

print("\n初始标签为：",end=" ")
for i in range(10):
    print(val_label[i],end=" ")
print()

plt.show()

sum = 0
for i in range(len(val_loader)):
    if val_label[i] == predict_label[i]:
        sum += 1
print("预测精度为：")
print(sum / len(val_loader))
