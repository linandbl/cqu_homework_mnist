import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import os
import struct
import gzip
import random
import time

# 配置参数
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.01
MOMENTUM = 0.9
PLOT_SAVE_PATH = "./model/mnist_cnn_ARAMATER_MANUL.png"
# 本地MNIST数据集
LOCAL_MNIST_DIR = "../cqu_homework/dataset/mnist/MNIST/raw"
# 采样比例
TRAIN_SAMPLE_RATIO = 1
TEST_SAMPLE_RATIO = 1
# 随机种子
RANDOM_SEED = 42

# 性能参数记录
train_loss_per_epoch = []
test_acc_per_epoch = []
test_f1_per_epoch = []
test_auc_per_epoch = []
epoch_list = []


# 自定义函数
def relu(x):
    """ReLU激活函数"""
    return np.maximum(0, x)


def relu_derivative(x):
    """ReLU导数"""
    return np.where(x > 0, 1, 0)


def softmax(x):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    """交叉熵损失"""
    n_samples = y_pred.shape[0]
    y_true_onehot = np.zeros_like(y_pred)
    y_true_onehot[np.arange(n_samples), y_true] = 1
    log_probs = -np.log(y_pred + 1e-10)  # 加小值避免log(0)
    loss = np.sum(log_probs * y_true_onehot) / n_samples
    return loss


def cross_entropy_derivative(y_pred, y_true):
    """交叉熵+Softmax联合导数"""
    n_samples = y_pred.shape[0]
    y_true_onehot = np.zeros_like(y_pred)
    y_true_onehot[np.arange(n_samples), y_true] = 1
    return (y_pred - y_true_onehot) / n_samples


def conv2d(x, kernel, padding=0):
    """
    2D卷积实现（batch, h, w, channels）
    """
    batch_size, in_h, in_w, in_channels = x.shape
    kernel_h, kernel_w, _, out_channels = kernel.shape

    # 填充
    if padding > 0:
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')

    out_h = (in_h + 2 * padding - kernel_h) + 1
    out_w = (in_w + 2 * padding - kernel_w) + 1

    output = np.zeros((batch_size, out_h, out_w, out_channels))

    for b in range(batch_size):
        for oc in range(out_channels):
            for h in range(out_h):
                for w in range(out_w):
                    receptive_field = x[b, h:h + kernel_h, w:w + kernel_w, :]
                    output[b, h, w, oc] = np.sum(receptive_field * kernel[:, :, :, oc])
    return output


def max_pool2d(x, pool_size=2, stride=2):
    """
    最大池化
    """
    batch_size, in_h, in_w, channels = x.shape
    out_h = (in_h - pool_size) // stride + 1
    out_w = (in_w - pool_size) // stride + 1

    output = np.zeros((batch_size, out_h, out_w, channels))
    mask = np.zeros_like(x)

    for b in range(batch_size):
        for c in range(channels):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * stride
                    h_end = h_start + pool_size
                    w_start = w * stride
                    w_end = w_start + pool_size

                    pool_region = x[b, h_start:h_end, w_start:w_end, c]
                    max_val = np.max(pool_region)
                    output[b, h, w, c] = max_val
                    max_pos = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                    mask[b, h_start + max_pos[0], w_start + max_pos[1], c] = 1
    return output, mask


def flatten(x):
    """展平"""
    return x.reshape(x.shape[0], -1)


def linear(x, weight, bias):
    """全连接"""
    return np.dot(x, weight) + bias


#进度条
def progress_bar(current, total, elapsed_time=0, bar_length=50):
    """
    终端进度条显示
    """
    progress = current / total
    percent = f"{progress * 100:.1f}%"
    filled_length = int(bar_length * progress)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    if current > 0 and elapsed_time > 0:
        speed = current / elapsed_time  # 每秒处理批次
        remaining_time = (total - current) / speed  # 剩余时间
        elapsed_str = f"{int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}"
        remaining_str = f"{int(remaining_time // 60):02d}:{int(remaining_time % 60):02d}"
        speed_str = f"{speed:.2f} batch/s"
        info = f"|{bar}| {current}/{total} {percent} | 已用: {elapsed_str} | 剩余: {remaining_str} | {speed_str}"
    else:
        info = f"|{bar}| {current}/{total} {percent} | 初始化中..."
    print(f"\r{info}", end="", flush=True)


# 数据采样
def random_sample_dataset(images, labels, sample_ratio, random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    total_samples = len(images)
    sample_num = int(total_samples * sample_ratio)
    sample_idx = np.random.choice(total_samples, sample_num, replace=False)
    sampled_images = images[sample_idx]
    sampled_labels = labels[sample_idx]
    return sampled_images, sampled_labels


# 本地MNIST数据加载
def check_mnist_files(data_dir):
    required_files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    missing_files = []
    for f in required_files:
        if not os.path.exists(os.path.join(data_dir, f)):
            missing_files.append(f)
    if missing_files:
        raise FileNotFoundError(
            f"本地MNIST数据集缺失以下文件：{missing_files}\n"
            f"请将文件放入目录：{data_dir}\n"
            f"文件来源：http://yann.lecun.com/exdb/mnist/"
        )


def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols, 1)
        images = images.astype(np.float32) / 255.0
        images = (images - 0.1307) / 0.3081
    return images


def load_mnist_labels(filename):
    """加载本地MNIST标签文件（gz压缩）"""
    with gzip.open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_local_mnist(data_dir):
    """加载本地MNIST数据集"""
    # 先校验文件是否存在
    check_mnist_files(data_dir)
    print(f"开始加载本地MNIST数据集：{data_dir}")
    # 加载训练集
    train_x = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    train_y = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    # 加载测试集
    test_x = load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    test_y = load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))
    print(f"原始数据集尺寸 | 训练集：{train_x.shape} | 测试集：{test_x.shape}")
    return (train_x, train_y), (test_x, test_y)


def create_data_loader(images, labels, batch_size, shuffle=True):
    dataset = list(zip(images, labels))
    total_batches = len(dataset) // batch_size + (1 if len(dataset) % batch_size != 0 else 0)

    def data_generator():
        if shuffle:
            random.shuffle(dataset)
        for i in range(0, len(dataset), batch_size):
            batch_imgs = np.array([item[0] for item in dataset[i:i + batch_size]])
            batch_lbls = np.array([item[1] for item in dataset[i:i + batch_size]])
            yield batch_imgs, batch_lbls

    return data_generator, total_batches


# CNN
class CNNModel:
    def __init__(self):
        self.conv1_weight = np.random.randn(5, 5, 1, 10) * 0.01  # 5*5*1→10
        self.conv1_bias = np.zeros(10)
        self.conv2_weight = np.random.randn(3, 3, 10, 20) * 0.01  # 3*3*10→20, padding=1
        self.conv2_bias = np.zeros(20)
        self.conv3_weight = np.random.randn(3, 3, 20, 20) * 0.01  # 3*3*20→20, padding=1
        self.conv3_bias = np.zeros(20)

        self.l1_weight = np.random.randn(180, 16) * 0.01  # 3*3*20=180 →16
        self.l1_bias = np.zeros(16)
        self.l2_weight = np.random.randn(16, 10) * 0.01  # 16→10（10分类）
        self.l2_bias = np.zeros(10)

        self.momentum = {k: np.zeros_like(getattr(self, k)) for k in [
            'conv1_weight', 'conv1_bias', 'conv2_weight', 'conv2_bias',
            'conv3_weight', 'conv3_bias', 'l1_weight', 'l1_bias',
            'l2_weight', 'l2_bias'
        ]}

        self.cache = {}

    def forward(self, x):
        self.cache['x'] = x
        # Conv1 → ReLU → MaxPool2d (28→24→12)
        conv1 = conv2d(x, self.conv1_weight) + self.conv1_bias
        relu1 = relu(conv1)
        pool1, mask1 = max_pool2d(relu1)
        self.cache.update({'conv1': conv1, 'relu1': relu1, 'pool1': pool1, 'mask1': mask1})

        # Conv2 → ReLU → MaxPool2d (12→12→6)  padding=1
        conv2 = conv2d(pool1, self.conv2_weight, padding=1) + self.conv2_bias
        relu2 = relu(conv2)
        pool2, mask2 = max_pool2d(relu2)
        self.cache.update({'conv2': conv2, 'relu2': relu2, 'pool2': pool2, 'mask2': mask2})

        # Conv3 → ReLU → MaxPool2d (6→6→3)  padding=1
        conv3 = conv2d(pool2, self.conv3_weight, padding=1) + self.conv3_bias
        relu3 = relu(conv3)
        pool3, mask3 = max_pool2d(relu3)
        self.cache.update({'conv3': conv3, 'relu3': relu3, 'pool3': pool3, 'mask3': mask3})

        # Flatten → Linear1 → ReLU → Linear2
        flat = flatten(pool3)
        fc1 = linear(flat, self.l1_weight, self.l1_bias)
        relu4 = relu(fc1)
        fc2 = linear(relu4, self.l2_weight, self.l2_bias)
        self.cache.update({'flat': flat, 'fc1': fc1, 'relu4': relu4, 'fc2': fc2})

        # Softmax输出
        output = softmax(fc2)
        self.cache['output'] = output
        return output

    def backward(self, y_true):
        batch_size = y_true.shape[0]

        d_fc2 = cross_entropy_derivative(self.cache['output'], y_true)

        d_l2_w = np.dot(self.cache['relu4'].T, d_fc2) / batch_size
        d_l2_b = np.sum(d_fc2, axis=0) / batch_size
        d_relu4 = np.dot(d_fc2, self.l2_weight.T)

        d_fc1 = d_relu4 * relu_derivative(self.cache['fc1'])
        d_l1_w = np.dot(self.cache['flat'].T, d_fc1) / batch_size
        d_l1_b = np.sum(d_fc1, axis=0) / batch_size
        d_flat = np.dot(d_fc1, self.l1_weight.T)

        d_pool3 = d_flat.reshape(self.cache['pool3'].shape)

        d_relu3 = np.zeros_like(self.cache['relu3'])
        for b in range(batch_size):
            for c in range(20):
                for h in range(3):
                    for w in range(3):
                        h_s, h_e = h * 2, h * 2 + 2
                        w_s, w_e = w * 2, w * 2 + 2
                        d_relu3[b, h_s:h_e, w_s:w_e, c] = self.cache['mask3'][b, h_s:h_e, w_s:w_e, c] * d_pool3[
                            b, h, w, c]
        d_conv3 = d_relu3 * relu_derivative(self.cache['conv3'])

        d_conv3_w = np.zeros_like(self.conv3_weight)
        pad_pool2 = np.pad(self.cache['pool2'], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant')
        for oc in range(20):
            for h in range(3):
                for w in range(3):
                    for ic in range(20):
                        d_conv3_w[h, w, ic, oc] = np.sum(
                            pad_pool2[:, h:h + 6, w:w + 6, ic] * d_conv3[:, :, :, oc]) / batch_size
        d_conv3_b = np.sum(d_conv3, axis=(0, 1, 2)) / batch_size

        d_relu2 = np.zeros_like(self.cache['relu2'])
        for b in range(batch_size):
            for c in range(20):
                for h in range(6):
                    for w in range(6):
                        h_s, h_e = h * 2, h * 2 + 2
                        w_s, w_e = w * 2, w * 2 + 2
                        d_relu2[b, h_s:h_e, w_s:w_e, c] = self.cache['mask2'][b, h_s:h_e, w_s:w_e, c] * d_conv3[
                            b, h, w, c]
        d_conv2 = d_relu2 * relu_derivative(self.cache['conv2'])

        d_conv2_w = np.zeros_like(self.conv2_weight)
        pad_pool1 = np.pad(self.cache['pool1'], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant')
        for oc in range(20):
            for h in range(3):
                for w in range(3):
                    for ic in range(10):
                        d_conv2_w[h, w, ic, oc] = np.sum(
                            pad_pool1[:, h:h + 12, w:w + 12, ic] * d_conv2[:, :, :, oc]) / batch_size
        d_conv2_b = np.sum(d_conv2, axis=(0, 1, 2)) / batch_size

        d_relu1 = np.zeros_like(self.cache['relu1'])
        for b in range(batch_size):
            for c in range(10):
                for h in range(12):
                    for w in range(12):
                        h_s, h_e = h * 2, h * 2 + 2
                        w_s, w_e = w * 2, w * 2 + 2
                        d_relu1[b, h_s:h_e, w_s:w_e, c] = self.cache['mask1'][b, h_s:h_e, w_s:w_e, c] * d_conv2[
                            b, h, w, c]
        d_conv1 = d_relu1 * relu_derivative(self.cache['conv1'])

        d_conv1_w = np.zeros_like(self.conv1_weight)
        for oc in range(10):
            for h in range(5):
                for w in range(5):
                    for ic in range(1):
                        d_conv1_w[h, w, ic, oc] = np.sum(
                            self.cache['x'][:, h:h + 24, w:w + 24, ic] * d_conv1[:, :, :, oc]) / batch_size
        d_conv1_b = np.sum(d_conv1, axis=(0, 1, 2)) / batch_size

        grads = {
            'conv1_weight': d_conv1_w, 'conv1_bias': d_conv1_b,
            'conv2_weight': d_conv2_w, 'conv2_bias': d_conv2_b,
            'conv3_weight': d_conv3_w, 'conv3_bias': d_conv3_b,
            'l1_weight': d_l1_w, 'l1_bias': d_l1_b,
            'l2_weight': d_l2_w, 'l2_bias': d_l2_b
        }
        return grads

    def update_parameters(self, grads, lr, momentum):
        for k in grads.keys():
            self.momentum[k] = momentum * self.momentum[k] - lr * grads[k]
            setattr(self, k, getattr(self, k) + self.momentum[k])


def train(model, train_loader, total_batches, epoch):
    total_loss = 0.0
    start_time = time.time()

    for batch_idx, (batch_imgs, batch_lbls) in enumerate(train_loader()):
        outputs = model.forward(batch_imgs)
        loss = cross_entropy_loss(outputs, batch_lbls)
        grads = model.backward(batch_lbls)
        model.update_parameters(grads, LEARNING_RATE, MOMENTUM)

        total_loss += loss
        # 计算已用时间，更新进度条
        elapsed_time = time.time() - start_time
        progress_bar(batch_idx + 1, total_batches, elapsed_time)

    print()
    avg_loss = total_loss / total_batches
    train_loss_per_epoch.append(avg_loss)
    total_time = time.time() - start_time
    print(f'Epoch {epoch + 1}/{EPOCHS} - 训练平均损失: {avg_loss:.6f} | 单轮耗时: {total_time:.2f}s')


def evaluate(model, test_loader, total_test_batches, epoch):
    correct = 0
    total = 0
    all_preds, all_labels, all_probs = [], [], []
    start_time = time.time()
    print(f"开始评估 Epoch {epoch + 1}...", end=" ")

    with np.errstate(all='ignore'):
        for batch_imgs, batch_lbls in test_loader():
            outputs = model.forward(batch_imgs)
            predicted = np.argmax(outputs, axis=1)
            total += batch_lbls.shape[0]
            correct += np.sum(predicted == batch_lbls)
            all_preds.extend(predicted)
            all_labels.extend(batch_lbls)
            all_probs.extend(outputs)

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
    # 打印评估结果
    eval_time = time.time() - start_time
    print(f"评估耗时: {eval_time:.2f}s")
    print(f'Epoch {epoch + 1}/{EPOCHS} - 测试准确率: {acc:.2f}% | F1: {f1:.4f} | AUC: {auc:.4f}\n')
    return acc, f1, auc

def plot_all_metrics():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    x = epoch_list
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MNIST模型训练性能指标', fontsize=16, fontweight='bold')

    # 训练损失
    ax1 = axes[0, 0]
    ax1.plot(x, train_loss_per_epoch, 'b-', marker='o', linewidth=2, markersize=6, label='训练损失')
    ax1.set_xlabel('Epoch'), ax1.set_ylabel('平均损失'), ax1.set_title('训练损失')
    ax1.legend(), ax1.grid(True, alpha=0.3, linestyle='--')
    for i, loss in enumerate(train_loss_per_epoch):
        ax1.annotate(f'{loss:.3f}', (x[i], loss), xytext=(0, 5), textcoords='offset points', ha='center')

    # 测试准确率
    ax2 = axes[0, 1]
    ax2.plot(x, test_acc_per_epoch, 'r-', marker='s', linewidth=2, markersize=6, label='测试准确率')
    ax2.set_xlabel('Epoch'), ax2.set_ylabel('准确率 (%)'), ax2.set_title('测试准确率')
    ax2.legend(), ax2.grid(True, alpha=0.3, linestyle='--')
    for i, acc in enumerate(test_acc_per_epoch):
        ax2.annotate(f'{acc:.1f}%', (x[i], acc), xytext=(0, 5), textcoords='offset points', ha='center')

    # F1分数
    ax3 = axes[1, 0]
    ax3.plot(x, test_f1_per_epoch, 'g-', marker='^', linewidth=2, markersize=6, label='Macro-F1')
    ax3.set_xlabel('Epoch'), ax3.set_ylabel('F1分数'), ax3.set_title('Macro-F1分数')
    ax3.legend(), ax3.grid(True, alpha=0.3, linestyle='--')
    for i, f1 in enumerate(test_f1_per_epoch):
        ax3.annotate(f'{f1:.3f}', (x[i], f1), xytext=(0, 5), textcoords='offset points', ha='center')

    # AUC分数
    ax4 = axes[1, 1]
    ax4.plot(x, test_auc_per_epoch, 'purple', marker='*', linewidth=2, markersize=8, label='ROC-AUC')
    ax4.set_xlabel('Epoch'), ax4.set_ylabel('AUC分数'), ax4.set_title('ROC-AUC分数')
    ax4.legend(), ax4.grid(True, alpha=0.3, linestyle='--')
    for i, auc in enumerate(test_auc_per_epoch):
        ax4.annotate(f'{auc:.3f}', (x[i], auc), xytext=(0, 5), textcoords='offset points', ha='center')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PLOT_SAVE_PATH, dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    try:
        (train_x, train_y), (test_x, test_y) = load_local_mnist(LOCAL_MNIST_DIR)

        print(f"\n开始按比例随机采样 | 训练集：{TRAIN_SAMPLE_RATIO * 100:.1f}% | 测试集：{TEST_SAMPLE_RATIO * 100:.1f}%")
        train_x_sampled, train_y_sampled = random_sample_dataset(
            train_x, train_y, TRAIN_SAMPLE_RATIO, RANDOM_SEED
        )
        test_x_sampled, test_y_sampled = random_sample_dataset(
            test_x, test_y, TEST_SAMPLE_RATIO, RANDOM_SEED
        )
        print(f"采样后数据集尺寸 | 训练集：{train_x_sampled.shape} | 测试集：{test_x_sampled.shape}")

        train_loader, train_total_batches = create_data_loader(train_x_sampled, train_y_sampled, BATCH_SIZE,
                                                               shuffle=True)
        test_loader, test_total_batches = create_data_loader(test_x_sampled, test_y_sampled, BATCH_SIZE, shuffle=False)

        model = CNNModel()

        print("\n===== 开始训练 =====")
        print(
            f"训练配置 | Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | 训练总批次: {train_total_batches} | 测试总批次: {test_total_batches}")
        for epoch in range(EPOCHS):
            train(model, train_loader, train_total_batches, epoch)
            evaluate(model, test_loader, test_total_batches, epoch)

        plot_all_metrics()
        print("===== 训练全部完成 =====")
    except FileNotFoundError as e:
        print(f"\n错误：{e}")
    except Exception as e:
        print(f"\n运行出错：{type(e).__name__} - {e}")