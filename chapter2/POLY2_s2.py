import torch
from torch import nn
from torch.nn import Parameter, init
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
from chapter2 import dataloader4ml100kOneHot  # 导入自定义的数据加载模块


# ==================== POLY2模型定义 ====================
class POLY2(nn.Module):
    """
    POLY2（多项式二阶）模型，用于CTR预估等二分类任务
    包含：线性部分（一阶特征） + 二阶交叉部分（特征两两交叉）
    公式: y = w0 + Σ(w1_i * x_i) + ΣΣ(w2_ij * x_i * x_j)
    """

    def __init__(self, n_features):
        """
        初始化POLY2模型参数

        :param n_features: 特征数量（输入特征维度）
        """
        super(POLY2, self).__init__()

        # 偏置项 w0: 形状 (1, 1)，使用Xavier均匀分布初始化
        # Parameter将张量包装为可训练参数，requires_grad自动为True
        self.w0 = init.xavier_uniform_(Parameter(torch.empty(1, 1)))

        # 一阶权重 w1: 形状 (n_features, 1)，用于线性部分
        # 每个特征对应一个权重
        self.w1 = init.xavier_uniform_(Parameter(torch.empty(n_features, 1)))

        # 二阶权重 w2: 形状 (n_features, n_features)，用于特征交叉
        # w2[i][j] 表示特征i和特征j交叉的权重
        self.w2 = init.xavier_uniform_(Parameter(torch.empty(n_features, n_features)))

    def crossLayer(self, x):
        """
        计算二阶特征交叉部分

        数学公式: ΣΣ(w2_ij * x_i * x_j)

        :param x: 输入特征，形状 [batch_size, n_features]
        :return: 二阶交叉结果，形状 [batch_size, 1]
        """
        # x_left: [batch_size, n_features, 1]
        # 在最后一维增加维度，用于矩阵乘法
        x_left = torch.unsqueeze(x, 2)

        # x_right: [batch_size, 1, n_features]
        # 在第一维增加维度，用于矩阵乘法
        x_right = torch.unsqueeze(x, 1)

        # x_cross: [batch_size, n_features, n_features]
        # 计算外积，得到所有特征两两组合的乘积
        # x_cross[i][j][k] = x[i][j] * x[i][k]
        x_cross = torch.matmul(x_left, x_right)

        # 加权求和：
        # 1. x_cross * self.w2: 逐元素相乘，得到每个交叉项的权重乘积
        # 2. torch.sum(..., dim=2): 对最后一个维度求和，形状变 [batch_size, n_features]
        # 3. torch.sum(..., dim=1, keepdim=True): 对特征维度求和，形状变 [batch_size, 1]
        cross_out = torch.sum(torch.sum(x_cross * self.w2, dim=2), dim=1, keepdim=True)
        return cross_out

    def forward(self, x):
        """
        前向传播：计算模型的预测输出

        :param x: 输入特征，形状 [batch_size, n_features]
        :return: 预测概率，形状 [batch_size, 1]，值在[0,1]之间
        """
        # 线性部分：偏置 + 一阶特征加权和
        # self.w0: [1,1], x: [batch_size, n_features], self.w1: [n_features, 1]
        # torch.matmul(x, self.w1): [batch_size, 1]
        lr_out = self.w0 + torch.matmul(x, self.w1)

        # 二阶交叉部分
        cross_out = self.crossLayer(x)

        # 使用sigmoid激活函数将输出映射到[0,1]区间
        # 用于二分类概率预测
        logits = torch.sigmoid(lr_out + cross_out)
        return logits


# ==================== 模型评估函数 ====================
def doEva(net, x, y):
    """
    评估模型在给定数据上的性能

    :param net: 训练好的模型
    :param x: 特征数据
    :param y: 真实标签
    :return: 准确率(precision)、召回率(recall)、精确度(accuracy)
    """
    # 将numpy数组转换为torch张量
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)

    # 禁用梯度计算，提高推理速度
    with torch.no_grad():
        out = net(x)  # 模型预测，输出为概率值

    # 将概率转换为类别（阈值0.5）
    # 大于等于0.5预测为正类(1)，否则为负类(0)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    y_true = y

    # 使用sklearn计算评估指标
    p = precision_score(y_true, y_pred)  # 精确率：预测为正类中实际为正类的比例
    r = recall_score(y_true, y_pred)  # 召回率：实际为正类中被正确预测的比例
    acc = accuracy_score(y_true, y_pred)  # 准确率：整体预测正确的比例

    return p, r, acc


# ==================== 训练函数 ====================
def train(epochs=20, batchSize=1024, lr=0.01, eva_per_epochs=1):
    """
    模型训练主函数

    :param epochs: 训练轮数
    :param batchSize: 批大小
    :param lr: 学习率
    :param eva_per_epochs: 每隔多少轮评估一次
    """

    # 1. 加载数据（从自定义模块读取）
    x_train, x_test, y_train, y_test = dataloader4ml100kOneHot.read_data()

    # 2. 获取特征维度（one-hot编码后的特征总数）
    features = len(x_train[0])

    # 3. 初始化POLY2模型
    net = POLY2(features)

    # 4. 定义损失函数：二元交叉熵损失（BCELoss）
    # 适用于二分类问题，配合sigmoid使用
    criterion = torch.nn.BCELoss()

    # 5. 创建数据迭代器（批数据生成器）
    dataIter = dataloader4ml100kOneHot.DataIter(x_train, y_train)

    # 6. 初始化优化器：AdamW（带权重衰减的Adam）
    # weight_decay: L2正则化系数，防止过拟合
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)

    # 7. 开始训练循环
    for e in range(epochs):
        all_lose = 0  # 累计当前epoch的总损失

        # 遍历所有批次数据
        for datas in dataIter.iter(batchSize=batchSize):
            # 提取当前批次的特征和标签
            Xs = torch.FloatTensor([d[0] for d in datas])  # 特征 [batch_size, n_features]
            labels = torch.FloatTensor([d[1] for d in datas])  # 标签 [batch_size]

            # 梯度清零（防止梯度累积）
            optimizer.zero_grad()

            # 前向传播：计算预测值
            logits = net(Xs)

            # 计算损失
            loss = criterion(logits, labels)
            all_lose += loss  # 累加损失

            # 反向传播：计算梯度
            loss.backward()

            # 更新模型参数
            optimizer.step()

        # 打印当前epoch的平均损失
        # 注意：这里用整除计算batch数量，可能不精确但足够用于监控
        print('epoch {}, avg_loss={:.4f}'.format(e, all_lose / (len(y_train) // batchSize)))

        # 定期评估模型性能
        if e % eva_per_epochs == 0:
            # 在训练集上评估
            p, r, acc = doEva(net, x_train, y_train)
            print('train:p:{:.4f}, r:{:.4f}, acc:{:.4f}'.format(p, r, acc))

            # 在测试集上评估
            p, r, acc = doEva(net, x_test, y_test)
            print('test:p:{:.4f}, r:{:.4f}, acc:{:.4f}'.format(p, r, acc))


# ==================== 程序入口 ====================
if __name__ == '__main__':
    # 开始训练模型
    # 参数：训练20轮，批大小1024，学习率0.01，每1轮评估一次
    train()