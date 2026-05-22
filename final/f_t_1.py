import numpy as np

# 给定数据：用户-物品评分矩阵
ratings = np.array([
    [5, 3, 0, 1, 0, 0, 4, 0],  # 用户1
    [4, 0, 0, 1, 2, 0, 0, 3],  # 用户2
    [1, 1, 0, 5, 0, 3, 0, 0],  # 用户3
    [0, 0, 4, 4, 4, 0, 1, 0],  # 用户4
    [0, 1, 5, 0, 1, 0, 0, 4],  # 用户5
    [3, 3, 0, 4, 2, 0, 0, 0],  # 用户6
], dtype=float)


# 任务1：计算两个用户评分向量之间的余弦相似度
def cosine_similarity(u, v):
    # 找到两人共同评分过的物品（即都不为 0 的维度）
    common_mask = (u > 0) & (v > 0)

    # 若无共同评分物品，返回 0
    if not np.any(common_mask):
        return 0.0

    u_common = u[common_mask]
    v_common = v[common_mask]

    # 计算余弦相似度
    dot_product = np.dot(u_common, v_common)
    norm_u = np.linalg.norm(u_common)
    norm_v = np.linalg.norm(v_common)

    if norm_u == 0 or norm_v == 0:
        return 0.0

    return dot_product / (norm_u * norm_v)


# 任务2：基于 UserCF 预测指定用户对指定物品的评分
def predict_rating(user_id, item_id, ratings, k=3):
    target_user = ratings[user_id]

    # 获取目标用户的评分均值 (仅计算已评分的物品)
    user_rated = target_user[target_user > 0]
    user_mean = np.mean(user_rated) if len(user_rated) > 0 else 0.0

    similarities = []
    # 遍历所有用户寻找邻居
    for other_user_id in range(len(ratings)):
        if other_user_id != user_id:
            # 候选邻居必须对目标物品有评分
            if ratings[other_user_id, item_id] > 0:
                sim = cosine_similarity(target_user, ratings[other_user_id])
                similarities.append((other_user_id, sim))

    # 按相似度降序排列，取 Top-K 邻居
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k_neighbors = similarities[:k]

    # 如果没有可用的邻居，直接返回用户的均值作为预测
    if not top_k_neighbors:
        return user_mean

    numerator = 0.0
    denominator = 0.0

    for neighbor_id, sim in top_k_neighbors:
        neighbor_ratings = ratings[neighbor_id]
        neighbor_rated = neighbor_ratings[neighbor_ratings > 0]
        neighbor_mean = np.mean(neighbor_rated)

        neighbor_item_rating = neighbor_ratings[item_id]

        numerator += sim * (neighbor_item_rating - neighbor_mean)
        denominator += abs(sim)

    if denominator == 0:
        return user_mean

    return user_mean + (numerator / denominator)


# 任务3：为目标用户推荐 Top-N 个物品
def recommend(user_id, ratings, top_n=3, k=3):
    target_user = ratings[user_id]
    # 找到该用户未评分的物品索引
    unrated_item_indices = np.where(target_user == 0)[0]

    predictions = []
    for item_id in unrated_item_indices:
        pred_score = predict_rating(user_id, item_id, ratings, k)
        predictions.append((item_id, pred_score))

    # 按预测评分降序排列，返回 Top-N
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_n]


# ---------------- 测试与格式化输出 ----------------
if __name__ == "__main__":
    num_users = len(ratings)

    # 1. 计算并打印用户相似度矩阵
    sim_matrix = np.zeros((num_users, num_users))
    for i in range(num_users):
        for j in range(num_users):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                sim_matrix[i, j] = cosine_similarity(ratings[i], ratings[j])

    print("=== 第一题：UserCF 推荐 ===")
    print(f"用户相似度矩阵（{num_users}x{num_users}）:")
    print("[", end="")
    for i in range(num_users):
        row_str = " ".join([f"{val:.4f}" for val in sim_matrix[i]])
        if i == 0:
            print(f"[{row_str}]")
        elif i == num_users - 1:
            print(f" [{row_str}]]\n")
        else:
            if i == 1:
                print(" ...")
            if i == num_users - 2:
                print(" ...")

    # 2. 计算用户1（索引0）对未评分物品的预测评分
    target_user_id = 0
    unrated_items = np.where(ratings[target_user_id] == 0)[0]

    for item_id in unrated_items:
        pred = predict_rating(target_user_id, item_id, ratings, k=3)
        print(f"用户1对物品{item_id + 1}的预测评分: {pred:.2f}")

    print(f"\n为用户1推荐 Top-3 物品:")
    # 3. 输出 Top-3 推荐结果
    top_recommendations = recommend(target_user_id, ratings, top_n=3, k=3)
    for item_id, pred_score in top_recommendations:
        print(f"物品{item_id + 1}, 预测评分: {pred_score:.2f}")