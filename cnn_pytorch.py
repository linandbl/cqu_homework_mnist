import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import os

# 训练超参数
BATCH_SIZE = 1024
EPOCHS = 20
DEVICE = torch.device("cpu")
MODEL_SAVE_PATH = "./model/mnist_cnn_model_pytorch.pth"  # 模型保存路径
PARAMATER_SAVE_PATH = "./model/mnist_cnn_PARAMATER_pytorch.png"  # 模型保存路径
print(f"使用设备: {DEVICE}")

# 性能参数记录
train_loss_per_epoch = []  # 每轮的平均训练损失
test_acc_per_epoch = []  # 每轮的测试准确率
test_f1_per_epoch = []  # 每轮的测试F1分数
test_auc_per_epoch = []  # 每轮的测试AUC分数
epoch_list = [] # 轮次

# 载入数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="../cqu_homework/dataset/mnist",
                               train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

test_dataset = datasets.MNIST(root="../cqu_homework/dataset/mnist", train=False,
                              download=False, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)


# CNN网络模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(20, 20, kernel_size=3, padding=1)
        self.pooling = torch.nn.MaxPool2d(2)
        self.l1 = torch.nn.Linear(180, 16)
        self.l2 = torch.nn.Linear(16, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.pooling(F.relu(self.conv2(x)))
        x = self.pooling(F.relu(self.conv3(x)))
        x = x.view(batch_size, -1)
        x = F.relu(self.l1(x))
        return self.l2(x)


# 实例化模型
model = Model().to(DEVICE)

# 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练
def train(epoch):
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs, target = inputs.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    # 计算本轮平均损失
    avg_loss = total_loss / total_batches
    train_loss_per_epoch.append(avg_loss)
    print(f'Epoch {epoch + 1}/{EPOCHS} - 训练平均损失: {avg_loss:.6f}')


# 测试
def evaluate(epoch):
    model.eval()
    correct = 0
    total = 0
    all_preds = []      # 每个类别的预测概率
    all_labels = [] # 每个类别的真实标签
    all_probs = []  # 每个类别的概率

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, dim=1)


            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算评估指标
    acc = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='macro')
    labels_bin = label_binarize(all_labels, classes=range(10))
    auc = roc_auc_score(labels_bin, np.array(all_probs), average='macro', multi_class='ovr')

    # 记录指标
    test_acc_per_epoch.append(acc)
    test_f1_per_epoch.append(f1)
    test_auc_per_epoch.append(auc)
    epoch_list.append(epoch + 1)

    # 打印本轮评估结果
    print(f'Epoch {epoch + 1}/{EPOCHS} - 测试准确率: {acc:.2f}% | F1分数: {f1:.4f} | AUC: {auc:.4f}')
    return acc, f1, auc


# 保存模型参数
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss_per_epoch,
        'test_acc': test_acc_per_epoch,
        'test_f1': test_f1_per_epoch,
        'test_auc': test_auc_per_epoch
    }, path)
    print(f"模型已保存至: {path}")

# 绘制性能指标图
def plot_all_metrics():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    x = epoch_list
    y_loss = train_loss_per_epoch    # 训练损失
    y_acc = test_acc_per_epoch       # 测试准确率
    y_f1 = test_f1_per_epoch         # F1分数
    y_auc = test_auc_per_epoch       # AUC分数

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 2行2列，14*10
    fig.suptitle('MNIST模型训练性能指标', fontsize=16, fontweight='bold')

    # loss
    ax1 = axes[0, 0]
    ax1.plot(x, y_loss, 'b-', marker='o', linewidth=2, markersize=6, label='训练损失')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('平均损失', fontsize=12)
    ax1.set_title('训练损失', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    for i, loss in enumerate(y_loss):
        ax1.annotate(f'{loss:.3f}', (x[i], loss), textcoords="offset points", xytext=(0, 5), ha='center')

    # acc
    ax2 = axes[0, 1]
    ax2.plot(x, y_acc, 'r-', marker='s', linewidth=2, markersize=6, label='测试准确率')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('准确率 (%)', fontsize=12)
    ax2.set_title('准确率', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    for i, acc in enumerate(y_acc):
        ax2.annotate(f'{acc:.1f}%', (x[i], acc), textcoords="offset points", xytext=(0, 5), ha='center')

    # F1
    ax3 = axes[1, 0]
    ax3.plot(x, y_f1, 'g-', marker='^', linewidth=2, markersize=6, label='Macro-F1分数')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('F1', fontsize=12)
    ax3.set_title('F1分数', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    for i, f1 in enumerate(y_f1):
        ax3.annotate(f'{f1:.3f}', (x[i], f1), textcoords="offset points", xytext=(0, 5), ha='center')

    # ROC-AUC
    ax4 = axes[1, 1]
    ax4.plot(x, y_auc, 'purple', marker='*', linewidth=2, markersize=8, label='ROC-AUC分数')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('AUC分数', fontsize=12)
    ax4.set_title('ROC-AUC分数', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, linestyle='--')
    for i, auc in enumerate(y_auc):
        ax4.annotate(f'{auc:.3f}', (x[i], auc), textcoords="offset points", xytext=(0, 5), ha='center')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PARAMATER_SAVE_PATH, dpi=600, bbox_inches='tight')
    plt.show()

# 主函数
if __name__ == '__main__':
    # 训练
    print("===== 开始训练 =====")
    for epoch in range(EPOCHS):
        train(epoch)
        evaluate(epoch)

    # 保存模型
    save_model(model, MODEL_SAVE_PATH)

    # 绘制指标图
    plot_all_metrics()
