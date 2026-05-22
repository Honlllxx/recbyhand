import numpy as np

# 给定数据：用户-物品评分矩阵 (8x6)
R = np.array([
    [5, 3, 0, 1, 0, 0],
    [4, 0, 0, 1, 0, 2],
    [1, 1, 0, 5, 0, 0],
    [0, 0, 4, 4, 4, 0],
    [0, 1, 5, 0, 0, 1],
    [3, 3, 0, 4, 0, 2],
    [0, 0, 5, 0, 5, 0],
    [2, 0, 3, 0, 1, 4],
], dtype=float)


# 任务1：训练 LFM 模型
def lfm_train(R, k, lr, lam, epochs):
    num_users, num_items = R.shape

    # 1. 初始化参数
    np.random.seed(42)  # 固定随机种子以保证结果可复现
    P = np.random.normal(0, 0.1, (num_users, k))
    Q = np.random.normal(0, 0.1, (num_items, k))
    bu = np.zeros(num_users)
    bi = np.zeros(num_items)

    # 计算全局均值 mu (只计算有评分的部分)
    non_zero_mask = R > 0
    mu = np.mean(R[non_zero_mask])

    # 获取所有有评分的用户和物品索引
    users, items = R.nonzero()

    # 2. 梯度下降迭代训练
    for epoch in range(epochs):
        # 遍历每一个非零样本进行随机梯度下降 (SGD)
        for u, i in zip(users, items):
            # 预测当前评分
            pred = mu + bu[u] + bi[i] + np.dot(P[u], Q[i])
            # 计算误差
            e = R[u, i] - pred

            # 更新偏置项
            bu[u] += lr * (e - lam * bu[u])
            bi[i] += lr * (e - lam * bi[i])

            # 更新隐向量 (注意：更新 Q[i] 时需要用到未更新前的 P[u])
            P_u = P[u].copy()
            P[u] += lr * (e * Q[i] - lam * P[u])
            Q[i] += lr * (e * P_u - lam * Q[i])

        # 每 100 轮打印一次 Loss，以及最后一轮
        if epoch % 100 == 0 or epoch == epochs - 1:
            loss = 0
            for u, i in zip(users, items):
                pred = mu + bu[u] + bi[i] + np.dot(P[u], Q[i])
                e = R[u, i] - pred
                # 损失函数 = 误差平方 + L2正则化项
                loss += e ** 2 + lam * (bu[u] ** 2 + bi[i] ** 2 + np.sum(P[u] ** 2) + np.sum(Q[i] ** 2))

            print(f"Epoch {epoch:4d}, Loss: {loss:7.4f}")

    return P, Q, bu, bi, mu


# 任务2：预测评分
def predict_lfm(user_id, item_id, P, Q, bu, bi, mu):
    # 根据公式计算预测值
    pred = mu + bu[user_id] + bi[item_id] + np.dot(P[user_id], Q[item_id])
    return pred


# 为用户推荐 Top-N 物品的辅助函数
def recommend_lfm(user_id, R, P, Q, bu, bi, mu, top_n=3):
    # 找到目标用户尚未评分的物品索引
    unrated_items = np.where(R[user_id] == 0)[0]

    predictions = []
    for item_id in unrated_items:
        pred_score = predict_lfm(user_id, item_id, P, Q, bu, bi, mu)
        predictions.append((item_id, pred_score))

    # 根据预测评分降序排序
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_n]


# ---------------- 任务3：测试与格式化输出 ----------------
if __name__ == "__main__":
    print("=== 第二题：LFM 矩阵分解 ===")

    # 1. 训练模型
    P, Q, bu, bi, mu = lfm_train(R, k=5, lr=0.005, lam=0.02, epochs=2000)

    # 2. 预测用户 1（索引 0）对物品 3（索引 2）和物品 5（索引 4）的评分
    target_user = 0
    item_3 = 2
    item_5 = 4

    pred_3 = predict_lfm(target_user, item_3, P, Q, bu, bi, mu)
    pred_5 = predict_lfm(target_user, item_5, P, Q, bu, bi, mu)

    print(f"\n用户1对物品3的预测评分: {pred_3:.4f}")
    print(f"用户1对物品5的预测评分: {pred_5:.4f}\n")

    # 3. 为用户 1 推荐 Top-3 物品
    print("为用户1推荐 Top-3 物品:")
    top_recommendations = recommend_lfm(target_user, R, P, Q, bu, bi, mu, top_n=3)

    for item_id, pred_score in top_recommendations:
        # 输出物品编号（索引+1）
        print(f"物品{item_id + 1}, 预测评分: {pred_score:.4f}")