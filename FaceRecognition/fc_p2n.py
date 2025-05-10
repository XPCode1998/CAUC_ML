import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import matplotlib.colors as mcolors


# 统计所有标签
def calLabels(img_path_list):
    # 姓名标签
    name_set = set()
    # 姿势标签
    pose_set = set()
    # 情绪标签
    mood_set = set()
    # 眼镜标签
    glasses_set = set()
    for img_path in img_path_list:
        file_name = os.path.basename(img_path)
        file_name_temp = file_name.split('.')[0]
        file_name_part = file_name_temp.split('_')
        name_temp = file_name_part[0]
        pose_temp = file_name_part[1]
        mood_temp = file_name_part[2]
        glasses_temp = file_name_part[3]
        name_set.add(name_temp)
        pose_set.add(pose_temp)
        mood_set.add(mood_temp)
        glasses_set.add(glasses_temp)

    # 创建标签字典
    name2id = dict((c, i) for i, c in enumerate(name_set))
    id2name = dict((v, k) for k, v in name2id.items())
    pose2id = dict((c, i) for i, c in enumerate(pose_set))
    id2pose = dict((v, k) for k, v in pose2id.items())
    mood2id = dict((c, i) for i, c in enumerate(mood_set))
    id2mood = dict((v, k) for k, v in mood2id.items())
    glasses2id = dict((c, i) for i, c in enumerate(glasses_set))
    id2glasses = dict((v, k) for k, v in glasses2id.items())

    return name2id, id2name, pose2id, id2pose, mood2id, id2mood, glasses2id, id2glasses


# 创建人脸数据集
class FaceDataset(Dataset):
    def __init__(self, img_path_list, name2id, pose2id, mood2id, glasses2id, transforms):
        self.img_path_list = img_path_list
        self.name2id = name2id
        self.pose2id = pose2id
        self.mood2id = mood2id
        self.glasses2id = glasses2id
        self.transforms = transforms.Compose([transforms.Resize((32, 30)), transforms.ToTensor()])

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        pil_img = Image.open(img_path)
        img_data = self.transforms(pil_img)

        file_name = os.path.basename(img_path)
        file_name_temp = file_name.split('.')[0]
        file_name_part = file_name_temp.split('_')
        name_label = self.name2id[file_name_part[0]]
        pose_label = self.pose2id[file_name_part[1]]
        mood_label = self.mood2id[file_name_part[2]]
        glasses_label = self.glasses2id[file_name_part[3]]
        return img_path, img_data, name_label, pose_label

    def __len__(self):
        return len(self.img_path_list)


# 人脸识别模型（标准版）
class FaceRecgNet(nn.Module):
    def __init__(self, name_num, pose_num):
        super(FaceRecgNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=7)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, name_num)
        self.fc3 = nn.Linear(128, pose_num)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        name = F.relu(self.fc2(x))
        pose = F.relu(self.fc3(x))
        return name, pose


class FaceRecgNet1(nn.Module):
    def __init__(self, name_num, pose_num):
        super(FaceRecgNet1, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=7)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(132, name_num)
        self.fc3 = nn.Linear(128, pose_num)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        pose = F.relu(self.fc3(x))
        x = torch.cat((x, pose), dim=1)
        name = F.relu(self.fc2(x))
        return name, pose


def train(epoch, device):
    running_loss = 0.0
    total_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        img_path, inputs, name_labels, pose_labels = data
        inputs, name_labels, pose_labels = inputs.to(device), name_labels.to(device), pose_labels.to(device)
        optimizer.zero_grad()
        name_outputs, pose_outputs = model(inputs)
        name_loss = criterion(name_outputs, name_labels)
        pose_loss = criterion(pose_outputs, pose_labels)
        loss = name_loss + pose_loss
        loss.backward()
        optimizer.step()
        # 用item，防止构建计算图
        running_loss += loss.item()
        total_loss += loss.item()
        if batch_idx % 20 == 19:
            print('[%d,%5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 20))
            running_loss = 0.0
    return total_loss / len(train_loader)


def test():
    name_correct = 0
    pose_correct = 0
    total = 0
    # 这里面的代码不会计算梯度
    with torch.no_grad():
        for data in test_loader:
            img_path, inputs, name_labels, pose_labels = data
            inputs, name_labels, pose_labels = inputs.to(device), name_labels.to(device), pose_labels.to(device)
            name_outputs, pose_outputs = model(inputs)
            # 找出最大值的下标
            # 行是第0个维度，从上至下
            # 列是第1个维度，从左至右
            _, name_predicted = torch.max(name_outputs.data, dim=1)
            _, pose_predicted = torch.max(pose_outputs.data, dim=1)
            total += name_outputs.size(0)
            # 张量之间的比较运算
            name_correct += (name_predicted == name_labels).sum().item()
            pose_correct += (pose_predicted == pose_labels).sum().item()
    print('name Accuracy on test set: %d %%' % (100 * name_correct / total))
    print('pose Accuracy on test set: %d %%' % (100 * pose_correct / total))
    return (name_correct / total), (pose_correct / total)


if __name__ == '__main__':
    # 图片地址列表
    img_path_list = glob.glob('afs/cs/project/theo-8/faceimages/faces/*/*.pgm')
    name2id, id2name, pose2id, id2pose, mood2id, id2mood, glasses2id, id2glasses = calLabels(
        img_path_list)

    # 计算标签数，用作线性分类层最后一层超参数
    name_num = len(name2id)
    pose_num = len(pose2id)
    mood_num = len(mood2id)
    glasses_num = len(glasses2id)

    BATCH_SIZE = 32
    img_train, img_test = train_test_split(img_path_list, test_size=0.2)
    train_dataset = FaceDataset(img_train, name2id, pose2id, mood2id, glasses2id, transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = FaceDataset(img_test, name2id, pose2id, mood2id, glasses2id, transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    epoch_num = 50
    learning_rate = 0.1
    device = torch.device('cpu')

    fig = plt.figure(figsize=(6.4, 14.4))
    # 设置不同的线型和灰度色彩
    line_styles = ['--', '-.', ':', '-']  # 不同的线型
    gray_levels = [0.2, 0.4, 0.6, 0.8]  # 不同的灰度色彩
    gray_colors = [mcolors.to_rgb((v, v, v)) for v in gray_levels]



    model = FaceRecgNet(name_num, pose_num)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    loss_list = []
    n_acc_list = []
    p_acc_list = []
    for epoch in range(epoch_num):
        start_time = time.time()
        loss = train(epoch, device)
        end_time = time.time()
        cost_time = end_time - start_time
        print('Training time %f m %f s' % (cost_time // 60, cost_time % 60))
        n_acc, p_acc = test()
        loss_list.append(loss)
        n_acc_list.append(n_acc)
        p_acc_list.append(p_acc)

    epochs = range(1, epoch_num + 1)
    plt.subplot(3, 1, 1)
    plt.plot(epochs, n_acc_list, linestyle=line_styles[0], color=gray_colors[0], label='pose+name')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(epochs, p_acc_list, linestyle=line_styles[0], color=gray_colors[0], label='pose+name')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(epochs, loss_list, linestyle=line_styles[0], color=gray_colors[0], label='pose+name')
    plt.legend()

    model = FaceRecgNet1(name_num, pose_num)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    loss_list = []
    n_acc_list = []
    p_acc_list = []
    for epoch in range(epoch_num):
        start_time = time.time()
        loss = train(epoch, device)
        end_time = time.time()
        cost_time = end_time - start_time
        print('Training time %f m %f s' % (cost_time // 60, cost_time % 60))
        n_acc, p_acc = test()
        loss_list.append(loss)
        n_acc_list.append(n_acc)
        p_acc_list.append(p_acc)

    epochs = range(1, epoch_num + 1)
    plt.subplot(3, 1, 1)
    plt.plot(epochs, n_acc_list, linestyle=line_styles[1], color=gray_colors[1], label='pose2name')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(epochs, p_acc_list, linestyle=line_styles[1], color=gray_colors[1], label='pose2name')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(epochs, loss_list, linestyle=line_styles[1], color=gray_colors[1], label='pose2name')
    plt.legend()

    plt.subplot(3, 1, 1)
    plt.title('Name Accuracy')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.title('Pose Accuracy')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.title('Training Loss')
    plt.legend()
    plt.show()
