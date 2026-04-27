from chapter2 import dataloader
from data_set import filepaths as fp
from tqdm import tqdm
from chapter2 import s2_basicSim as b_sim
import collections
import pickle  # 用来保存/加载模型

#   1. 原有的模型函数   
def getSet(triples):
    user_items = collections.defaultdict(set)
    for u, i, r in triples:
        if r == 1:
            user_items[u].add(i)
    return user_items

def knn4set(trainset, k, sim_method):
    sims = {}
    for e1 in tqdm(trainset):
        ulist = []
        for e2 in trainset:
            if e1 == e2 or len(trainset[e1] & trainset[e2]) == 0:
                continue
            sim = sim_method(trainset[e1], trainset[e2])
            ulist.append((e2, sim))
        sims[e1] = [i[0] for i in sorted(ulist, key=lambda x:x[1], reverse=True)[:k]]
    return sims

def get_recomedations_by_usrCF(user_sims, user_o_set):
    recomedations = collections.defaultdict(set)
    for u in user_sims:
        for sim_u in user_sims[u]:
            recomedations[u] |= (user_o_set[sim_u] - user_o_set[u])
    return recomedations

def trainUserCF(user_items, sim_method, k=5):
    user_sims = knn4set(user_items, k, sim_method)
    recomedations = get_recomedations_by_usrCF(user_sims, user_items)
    return user_sims, recomedations  # 返回 相似用户表 + 推荐结果


#   2. 保存模型（训练好存起来）   
def save_model(model, path="userCF_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"模型已保存到：{path}")

#   3. 加载模型（下次直接用，不用重新训练）   
def load_model(path="userCF_model.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"模型已加载：{path}")
    return model


#   4. 使用模型：给某个用户推荐物品   
def recommend_for_user(user_id, user_sims, user_items, topN=10):
    """
    给指定用户推荐 TopN 物品
    """
    if user_id not in user_sims:
        return []  # 用户没历史，无法推荐

    rec_items = set()
    for sim_user in user_sims[user_id]:
        rec_items.update(user_items[sim_user])

    # 去掉用户已经喜欢过的
    rec_items -= user_items[user_id]

    # 转成列表返回
    return list(rec_items)[:topN]


#   主程序：训练 + 使用   
if __name__ == '__main__':
    # ========== 步骤1：读取数据 ==========
    _, _, train_set, test_set = dataloader.readRecData(fp.Ml_100K.RATING, test_ratio=0.1)
    user_items = getSet(train_set)

    # ========== 步骤2：训练模型 ==========
    print("开始训练 UserCF 模型...")
    user_sims, rec_result = trainUserCF(user_items, b_sim.cos4set, k=5)

    # ========== 步骤3：保存模型（以后直接加载） ==========
    save_model(
        {
            "user_sims": user_sims,  # 关键：用户相似度表
            "user_items": user_items  # 关键：用户历史喜欢物品
        },
        path="userCF_model.pkl"
    )

    # ========== 步骤4：加载模型（真实使用时，只运行这一步） ==========
    model = load_model("userCF_model.pkl")
    user_sims = model["user_sims"]
    user_items = model["user_items"]

    # ========== 步骤5：给用户做推荐！！！ ==========
    test_user_id = 1  # 选一个用户 ID
    rec_list = recommend_for_user(test_user_id, user_sims, user_items, topN=10)

    print(f"\n给用户 {test_user_id} 的推荐结果：")
    print(rec_list)