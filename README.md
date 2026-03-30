# 任务记录

**日期：** 2026年3月30日

## 环境配置

```bash
pip install -r requirements.txt
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
```

## 任务一：双塔模型的 ALS+MLP 算法

### 文件信息

- **文件路径：** `recbyhand/chapter3/s11a_ALS_MLP.py`
- **类名：** `ALS_MLP`

### 实现特点

- 用户和物品分别通过独立的 MLP 塔进行处理
- 两个塔的输出进行内积计算得到预测结果
- 典型的双塔架构设计

------

## 任务二：先拼接后传播的 ALS+MLP 算法

### 文件信息

- **文件路径：** `recbyhand/chapter3/s11b_ALS_CONCAT.py`
- **类名：** `ALS_MLP`

### 实现特点

- 先将用户和物品的嵌入向量拼接
- 拼接后的向量通过统一的 MLP 进行处理
- 单塔架构，先拼接再传播

------

## 任务三：特征向量 + MLP 算法

### 文件信息

- **文件路径：** `recbyhand/chapter3/s12_Embedding_MLP.py`
- **类名：** `embedding_mlp`

### 实现特点

- 使用嵌入层为所有用户和物品特征生成特征向量
- 将所有特征向量拼接后通过 MLP 进行处理
- 支持基于特征向量的推荐模型