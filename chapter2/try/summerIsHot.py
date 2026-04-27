import json
import math
from sklearn.metrics import roc_auc_score, accuracy_score

# 1. 加载测试集
with open("test_set.json", "r", encoding="utf-8") as f:
    data = json.load(f)

  
# 任务1 UserCF/ItemCF 评估
  
true_rating = [d["评分"] for d in data["任务1_协同过滤(UserCF/ItemCF)"]]
pred_rating = [5,3,4,2,5,1]  # 替换成你的模型预测结果

mse = sum((t-p)**2 for t,p in zip(true_rating, pred_rating)) / len(true_rating)
rmse = math.sqrt(mse)
print("UserCF/ItemCF RMSE:", rmse)

  
# 任务2 ALS 评估
  
true_rating_als = [d["评分"] for d in data["任务2_ALS矩阵分解"]]
pred_rating_als = [5,2,3,4,5]  # 替换成你的预测

mse_als = sum((t-p)**2 for t,p in zip(true_rating_als, pred_rating_als)) / len(true_rating_als)
rmse_als = math.sqrt(mse_als)
print("ALS RMSE:", rmse_als)

  
# 任务3 FM 评估
  
true_label = [d["标签"] for d in data["任务3_FM因子分解机"]]
pred_prob = [0.9, 0.1, 0.8, 0.2, 0.7]  # 替换成FM输出的概率
pred_label = [1 if p>0.5 else 0 for p in pred_prob]

acc = accuracy_score(true_label, pred_label)
auc = roc_auc_score(true_label, pred_prob)
print("FM 准确率:", acc)
print("FM AUC:", auc)