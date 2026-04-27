from chapter2 import dataloader
from data_set import filepaths as fp
from tqdm import tqdm
from chapter2 import s2_basicSim as b_sim
import collections

# 通用数据构建
def get_set_map(triples):
    user_items = collections.defaultdict(set)
    item_users = collections.defaultdict(set)
    for u, i, r in triples:
        if r == 1:
            user_items[u].add(i)
            item_users[i].add(u)
    return user_items, item_users

# 通用 KNN
def knn(train_map, k, sim_method):
    sim_dic = {}
    for a in tqdm(train_map):
        neighbors = []
        for b in train_map:
            if a == b or not train_map[a] & train_map[b]:
                continue
            neighbors.append((b, sim_method(train_map[a], train_map[b])))
        sim_dic[a] = [x[0] for x in sorted(neighbors, key=lambda x: x[1], reverse=True)[:k]]
    return sim_dic

# 通用推荐引擎
class CFModel:
    def __init__(self, kind='userCF', k=5):
        self.kind = kind
        self.k = k

    def train(self, user_items, item_users, sim_func):
        if self.kind == 'userCF':
            sims = knn(user_items, self.k, sim_func)
            rec = self._userCF_rec(sims, user_items)
        else:
            sims = knn(item_users, self.k, sim_func)
            rec = self._itemCF_rec(sims, user_items)
        return rec

    def _userCF_rec(self, user_sims, user_items):
        rec = collections.defaultdict(set)
        for u in user_sims:
            for v in user_sims[u]:
                rec[u] |= user_items[v] - user_items[u]
        return rec

    def _itemCF_rec(self, item_sims, user_items):
        rec = collections.defaultdict(set)
        for u, items in user_items.items():
            for i in items:
                if i in item_sims:
                    rec[u] |= set(item_sims[i]) - items
        return rec

if __name__ == '__main__':
    _, _, train, test = dataloader.readRecData(fp.Ml_100K.RATING, 0.1)
    user_items, item_users = get_set_map(train)

    # 训练 UserCF
    model1 = CFModel(kind='userCF', k=5)
    rec1 = model1.train(user_items, item_users, b_sim.cos4set)

    # 训练 ItemCF
    model2 = CFModel(kind='itemCF', k=5)
    rec2 = model2.train(user_items, item_users, b_sim.cos4set)

    print("UserCF 推荐结果:\n", rec1)
    print("ItemCF 推荐结果:\n", rec2)