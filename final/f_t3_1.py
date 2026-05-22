import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# ========== 给定数据 ==========
hist_seq = np.array([
    [0, 2, 4, 7], [1, 3, 6, 5], [0, 4, 5, 7], [2, 5, 1, 3],
    [1, 2, 3, 5], [6, 0, 2, 4], [0, 1, 2, 4], [3, 5, 6, 7],
    [7, 0, 3, 2], [0, 3, 6, 7], [4, 5, 1, 2], [1, 2, 4, 6]
])
target_items = np.array([5, 0, 3, 6, 4, 1, 7, 2, 5, 2, 0, 3])
ratings = np.array([4, 2, 5, 3, 4, 1, 5, 3, 2, 4, 3, 5])


# ===== 任务1：数据预处理函数 =====
def build_dataset(hist_seq, target_items, ratings):
    hist_tensor = torch.LongTensor(hist_seq)
    target_tensor = torch.LongTensor(target_items)
    ratings_tensor = torch.FloatTensor(ratings)

    # 统计物品数量，由于 ID 从 0 开始，最大值 + 1 即为总数
    num_items = int(np.max(hist_seq) + 1)

    return hist_tensor, target_tensor, ratings_tensor, num_items


# ===== 任务2：定义序列推荐模型类 =====
class SequenceModel(nn.Module):
    def __init__(self, num_items, d, h_dim):
        super(SequenceModel, self).__init__()
        # 定义物品 Embedding 层
        self.item_embedding = nn.Embedding(num_items, d)

        # 定义多层感知机 (MLP)
        # 输入维度为 2*d (用户兴趣向量维度 d + 目标物品向量维度 d)
        self.fc1 = nn.Linear(2 * d, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.out = nn.Linear(h_dim, 1)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, hist_seq, target_items):
        # 1. 查表得到 Embedding
        hist_emb = self.item_embedding(hist_seq)  # (N, L, d)
        target_emb = self.item_embedding(target_items)  # (N, d)

        # 2. 沿序列维度 L 进行求和池化得到用户兴趣向量
        user_vec = torch.sum(hist_emb, dim=1)  # (N, d)

        # 3. 沿特征维度拼接
        concat = torch.cat([user_vec, target_emb], dim=1)  # (N, 2d)

        # 4. 前向过全连接层加激活函数
        h1 = self.relu(self.fc1(concat))  # (N, h_dim)
        h2 = self.relu(self.fc2(h1))  # (N, h_dim)

        # 5. 输出层并去除末维
        pred = self.out(h2).squeeze(-1)  # (N,)

        # 6. 将预测值裁剪到 1.0 ~ 5.0 之间
        pred_ratings = torch.clamp(pred, min=1.0, max=5.0)

        return pred_ratings


# ===== 任务3：编写训练函数 =====
def train_sequence_model(model, hist_tensor, target_tensor, ratings_tensor, lr, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        predictions = model(hist_tensor, target_tensor)
        # 计算损失
        loss = criterion(predictions, ratings_tensor)

        # 反向传播与优化
        loss.backward()
        optimizer.step()

        # 每 50 轮或最后一轮打印一次 Loss
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}, Loss: {loss.item():7.4f}")

    return model


# ===== 任务4：模型评估与主流程 =====
if __name__ == "__main__":
    # 转换数据
    hist_tensor, target_tensor, ratings_tensor, num_items = build_dataset(hist_seq, target_items, ratings)

    # 超参数定义
    d = 8
    h_dim = 16
    lr = 0.01
    epochs = 500

    print("=== 第三题：基本序列推荐模型 ===")
    print(f"数据集: {hist_tensor.shape[0]} 个样本, {num_items} 个物品, Embedding 维度 d={d}, 隐藏层维度={h_dim}\n")

    # 初始化模型并训练
    model = SequenceModel(num_items, d, h_dim)
    model = train_sequence_model(model, hist_tensor, target_tensor, ratings_tensor, lr, epochs)

    print("\n训练完成！")

    # 模型评估
    model.eval()
    with torch.no_grad():
        final_predictions = model(hist_tensor, target_tensor)

    # 计算评估指标 (RMSE & MAE)
    final_predictions_np = final_predictions.numpy()
    ratings_np = ratings_tensor.numpy()

    rmse = np.sqrt(np.mean((final_predictions_np - ratings_np) ** 2))
    mae = np.mean(np.abs(final_predictions_np - ratings_np))

    # 按要求打印对比结果
    print("\n==================== 模型评估 ====================")
    print(f"{'样本':<5} | {'预测评分':<8} | {'真实评分':<8}")
    print("-" * 40)
    for i in range(5):
        print(f"用户{i + 1:<2}  |  {final_predictions_np[i]:.2f}    |  {ratings_np[i]:.2f}")
    print("-" * 40)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")