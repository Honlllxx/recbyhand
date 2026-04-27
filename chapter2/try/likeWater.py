import math
import collections
from tqdm import tqdm
from chapter2 import dataloader
from data_set import filepaths as fp


def getSet(triples):
    """
    只需构建用户喜欢的物品集合。
    因为使用共现矩阵法，不再需要专门维护 item_users。
    """
    user_items = collections.defaultdict(set)
    for u, i, r in triples:
        if r == 1:
            user_items[u].add(i)
    return user_items


def calculate_item_similarity(user_items, k=20):
    '''
    核心优化：使用“共现矩阵”计算物品相似度（余弦相似度）
    :param user_items: 用户喜欢的物品集合
    :param k: 为每个物品保留的最相似的Top-K个物品
    :return: {物品1: [(相似物品1, 分数), (相似物品2, 分数)...]}
    '''
    # 1. 统计物品热度 & 构建物品共现矩阵
    item_co_occurrence = collections.defaultdict(lambda: collections.defaultdict(int))
    item_popularities = collections.defaultdict(int)

    for user, items in tqdm(user_items.items(), desc="Building co-occurrence matrix"):
        for i in items:
            item_popularities[i] += 1
            for j in items:
                if i == j:
                    continue
                # 用户同时喜欢物品 i 和物品 j，共现次数 +1
                item_co_occurrence[i][j] += 1

    # 2. 计算余弦相似度并保留 Top-K
    item_sims = {}
    for i, related_items in tqdm(item_co_occurrence.items(), desc="Calculating Item Similarities"):
        sims_for_i = []
        for j, co_times in related_items.items():
            # 余弦相似度公式: |N(i) ∩ N(j)| / sqrt(|N(i)| * |N(j)|)
            sim_score = co_times / math.sqrt(item_popularities[i] * item_popularities[j])
            sims_for_i.append((j, sim_score))

        # 按相似度降序排序，只保留 Top-K 的相似物品
        item_sims[i] = sorted(sims_for_i, key=lambda x: x[1], reverse=True)[:k]

    return item_sims


def get_recommendations_by_itemCF(item_sims, user_items, top_n=10):
    '''
    优化：基于相似度得分进行累加排序，给出Top-N推荐
    :param item_sims: 物品相似表 {物品: [(相似物品, 相似度)...]}
    :param user_items: 用户历史喜欢物品
    :param top_n: 为每个用户推荐的物品数量
    :return: 推荐结果 {用户: [推荐物品1, 推荐物品2...]}
    '''
    recommendations = {}

    for u, items in tqdm(user_items.items(), desc="Generating Recommendations"):
        rank = collections.defaultdict(float)

        # 遍历用户喜欢的每一个物品
        for i in items:
            if i not in item_sims:
                continue

            # 遍历与物品 i 最相似的 k 个物品
            for j, sim_score in item_sims[i]:
                if j in items:  # 排除用户已经喜欢过的物品
                    continue
                # 累加推荐得分：假设用户历史行为权重均为 1，得分即为相似度累加
                rank[j] += sim_score

        # 取出得分最高的 Top-N 个物品作为推荐结果
        recommendations[u] = [item for item, score in sorted(rank.items(), key=lambda x: x[1], reverse=True)[:top_n]]

    return recommendations


def trainItemCF(user_items, k=20, top_n=10):
    # 1. 计算物品相似度矩阵 (K近邻)
    item_sims = calculate_item_similarity(user_items, k=k)
    # 2. 生成推荐列表 (Top-N)
    recommendations = get_recommendations_by_itemCF(item_sims, user_items, top_n=top_n)
    return recommendations


if __name__ == '__main__':
    # 读取数据
    _, _, train_set, test_set = dataloader.readRecData(fp.Ml_100K.RATING, test_ratio=0.1)

    # 构建用户-物品集合
    user_items = getSet(train_set)

    # 训练 ItemCF 模型
    # k: 相似物品的数量池 (通常 10-50), top_n: 最终推荐给用户的数量
    recom_by_itemCF = trainItemCF(user_items, k=20, top_n=10)

    # 输出结果示例（打印前5个用户的推荐结果）
    sample_users = list(recom_by_itemCF.keys())[:5]
    for u in sample_users:
        print(f"User {u}: {recom_by_itemCF[u]}")