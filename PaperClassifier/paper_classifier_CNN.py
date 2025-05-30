import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.optim as optim
from load_data import load_data
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence


class PaperDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def collata_fn(examples):
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, targets





class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, filter_size, num_filter, num_class):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1d = nn.Conv1d(embedding_dim, num_filter, filter_size, padding=1)
        self.activate = F.relu
        self.linear = nn.Linear(num_filter, num_class)

    def forward(self, inputs):
        embedding = self.embedding(inputs)
        convolution = self.activate(self.conv1d(embedding.permute(0, 2, 1)))
        pooling = F.max_pool1d(convolution, kernel_size=convolution.shape[2])
        outputs = self.linear(pooling.squeeze(dim=2))
        log_probs = F.log_softmax(outputs, dim=1)
        return log_probs


def train(epoch):
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f'Training Epoch {epoch}'):
        optimizer.zero_grad()
        inputs, targets = [x.to(device) for x in batch]
        log_probs = model(inputs)
        loss = criterion(log_probs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss = total_loss / len(train_dataset)
    print(f'Loss:{total_loss:.4f}')
    return total_loss


def test():
    acc = 0
    for batch in tqdm(test_dataloader, desc=f'Testing'):
        inputs, targets = [x.to(device) for x in batch]
        with torch.no_grad():
            output = model(inputs)
            acc += (output.argmax(dim=1) == targets).sum().item()
    print(f'Acc:{acc * 100 / len(test_dataset):.4f}%')
    return acc / len(test_dataset)




if __name__ == '__main__':
    embedding_dim = 128
    hidden_dim = 256
    class_num = 2
    batch_size = 32
    epoch_num = 100
    filter_size = 3
    num_filter = 100
    learning_rate=0.001

    train_data, test_data, vocab = load_data()
    train_dataset = PaperDataset(train_data)
    test_dataset = PaperDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collata_fn, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collata_fn, shuffle=True)

    device = torch.device('cpu')
    model = CNN(len(vocab), embedding_dim, filter_size, num_filter, class_num)
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_list = []
    acc_list = []
    for epoch in range(epoch_num):
        loss = train(epoch)
        acc = test()
        loss_list.append(loss)
        acc_list.append(acc)

    fig = plt.figure(figsize=(12.8, 4.8))
    # 设置不同的线型和灰度色彩
    line_styles = ['--', '-.', ':', '-']  # 不同的线型
    gray_levels = [0.2, 0.4, 0.6, 0.8]  # 不同的灰度色彩
    gray_colors = [mcolors.to_rgb((v, v, v)) for v in gray_levels]
    # 画出训练的损失与测试的准确率
    epochs = range(1, epoch_num + 1)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc_list, linestyle=line_styles[0], color=gray_colors[0], ')
    plt.legend()
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss_list, linestyle=line_styles[1], color=gray_colors[1], label='Training Loss')
    plt.legend()
    plt.title('Loss')
    plt.legend()

    plt.show()



